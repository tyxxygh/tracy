#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

#include "TracyFilesystem.hpp"
#include "TracyImGui.hpp"
#include "TracyPrint.hpp"
#include "TracyView.hpp"
#include "tracy_pdqsort.h"
#include "../Fonts.hpp"

namespace tracy
{

#ifndef TRACY_NO_STATISTICS

// Aggregate, per source location, the time spent inside two frame time ranges (A and B),
// then diff them so the user can see which zones explain the per-frame variation.
void View::ComputeFrameCompare()
{
    auto& fc = m_frameCompare;
    fc.dirty = false;
    fc.results.clear();
    fc.highlightSrcLoc.clear();
    fc.highlightZonesCpu.clear();
    fc.highlightZonesGpu.clear();
    fc.frameTimeA = 0;
    fc.frameTimeB = 0;

    // Per source location, the specific zone instances that fall inside frame A or B.
    // Only these instances are highlighted on the timeline (not every same-named zone in the trace).
    unordered_flat_map<int16_t, std::vector<const void*>> inRangePtrs;

    const FrameData* fd = fc.frameSet ? fc.frameSet : m_worker.GetFramesBase();
    if( !fd ) return;
    const int cnt = (int)m_worker.GetFrameCount( *fd );
    if( fc.frameA < 0 || fc.frameA >= cnt || fc.frameB < 0 || fc.frameB >= cnt ) return;

    const auto minA = m_worker.GetFrameBegin( *fd, fc.frameA );
    const auto maxA = m_worker.GetFrameEnd( *fd, fc.frameA );
    const auto minB = m_worker.GetFrameBegin( *fd, fc.frameB );
    const auto maxB = m_worker.GetFrameEnd( *fd, fc.frameB );
    fc.frameTimeA = maxA - minA;
    fc.frameTimeB = maxB - minB;

    const bool gpu = fc.mode == 0;
    fc.resultsAreGpu = gpu;

    if( gpu )
    {
        if( !m_worker.AreGpuSourceLocationZonesReady() ) return;

        // Walk the GPU timelines exactly the way they are drawn: per GPU context, per thread,
        // each zone converted to CPU-aligned time with that context/thread's own begin + drift.
        // This avoids a thread-id-keyed calibration map, which would be wrong when one submitting
        // thread feeds several GPU contexts (e.g. main pass and shadow pass on different queues).
        unordered_flat_map<int16_t, FrameCompareData::DiffEntry> gpuAgg;

        std::function<void( const Vector<short_ptr<GpuEvent>>&, int64_t, int )> walk =
            [&]( const Vector<short_ptr<GpuEvent>>& vec, int64_t begin, int drift )
        {
            auto process = [&]( const GpuEvent& ev )
            {
                const auto start = AdjustGpuTime( ev.GpuStart(), begin, drift );
                const auto end   = AdjustGpuTime( ev.GpuEnd(),   begin, drift );
                const auto sl = ev.SrcLoc();
                // GPU work for a frame does not fit inside the CPU frame markers (a GPU-bound
                // frame's top GPU zone is longer than the CPU frame window), so containment fails;
                // but plain overlap double-counts a big zone into both windows when A and B are
                // adjacent. Assign each zone to exactly one frame by its midpoint, and report its
                // full duration. The midpoint lands in the frame the zone is drawn under.
                const auto mid = start + ( end - start ) / 2;
                if( mid >= minA && mid < maxA )
                {
                    auto& e = gpuAgg[sl];
                    e.srcloc = sl;
                    e.timeA += end - start;
                    e.cntA++;
                    inRangePtrs[sl].emplace_back( &ev );
                }
                else if( mid >= minB && mid < maxB )
                {
                    auto& e = gpuAgg[sl];
                    e.srcloc = sl;
                    e.timeB += end - start;
                    e.cntB++;
                    inRangePtrs[sl].emplace_back( &ev );
                }
                if( ev.Child() >= 0 ) walk( m_worker.GetGpuChildren( ev.Child() ), begin, drift );
            };

            if( vec.is_magic() )
            {
                auto& v = *(Vector<GpuEvent>*)&vec;
                for( auto& ev : v ) process( ev );
            }
            else
            {
                for( auto& evp : vec ) process( *evp );
            }
        };

        for( auto& ctx : m_worker.GetGpuData() )
        {
            const int drift = GpuDrift( ctx );
            for( auto& td : ctx->threadData )
            {
                auto& tl = td.second.timeline;
                if( tl.empty() ) continue;
                int64_t begin;
                if( tl.is_magic() )
                    begin = ((const Vector<GpuEvent>*)&tl)->front().GpuStart();
                else
                    begin = tl.front()->GpuStart();
                if( begin < 0 ) continue;
                walk( tl, begin, drift );
            }
        }

        for( auto& kv : gpuAgg )
        {
            if( kv.second.cntA != 0 || kv.second.cntB != 0 ) fc.results.emplace_back( kv.second );
        }
    }
    else
    {
        if( !m_worker.AreSourceLocationZonesReady() ) return;

        const auto accMode = fc.accumulationMode;
        const auto rangeEnd = std::max( maxA, maxB );
        auto& slz = m_worker.GetSourceLocationZones();
        for( auto it = slz.begin(); it != slz.end(); ++it )
        {
            int64_t totalA = 0, totalB = 0;
            uint32_t cntA = 0, cntB = 0;
            for( auto& v : it->second.zones )
            {
                auto& z = *v.Zone();
                const auto start = z.Start();
                if( start > rangeEnd ) break;   // zones are sorted by start time
                const auto end = z.End();
                const auto zt = end - start;
                if( start >= minA && end <= maxA )
                {
                    if( accMode == AccumulationMode::SelfOnly )
                    {
                        totalA += zt - GetZoneChildTimeFast( z );
                        cntA++;
                        inRangePtrs[it->first].emplace_back( &z );
                    }
                    else if( accMode == AccumulationMode::AllChildren || !IsZoneReentry( z ) )
                    {
                        totalA += zt;
                        cntA++;
                        inRangePtrs[it->first].emplace_back( &z );
                    }
                }
                else if( start >= minB && end <= maxB )
                {
                    if( accMode == AccumulationMode::SelfOnly )
                    {
                        totalB += zt - GetZoneChildTimeFast( z );
                        cntB++;
                        inRangePtrs[it->first].emplace_back( &z );
                    }
                    else if( accMode == AccumulationMode::AllChildren || !IsZoneReentry( z ) )
                    {
                        totalB += zt;
                        cntB++;
                        inRangePtrs[it->first].emplace_back( &z );
                    }
                }
            }
            if( cntA != 0 || cntB != 0 )
            {
                fc.results.emplace_back( FrameCompareData::DiffEntry { it->first, totalA, totalB, cntA, cntB } );
            }
        }
    }

    // Sort by absolute time difference, biggest contributors to the variation first.
    pdqsort_branchless( fc.results.begin(), fc.results.end(), []( const auto& lhs, const auto& rhs ) {
        return llabs( lhs.timeA - lhs.timeB ) > llabs( rhs.timeA - rhs.timeB );
    } );

    // Mark source locations whose delta is a meaningful fraction of the frame-time delta,
    // then highlight only their zone instances that actually live inside frame A or B.
    const auto frameDelta = llabs( fc.frameTimeA - fc.frameTimeB );
    const auto ref = frameDelta != 0 ? frameDelta : std::max( fc.frameTimeA, fc.frameTimeB );
    if( ref != 0 )
    {
        const auto thr = (int64_t)( ref * ( fc.highlightThreshold / 100.0 ) );
        auto& zoneSet = gpu ? fc.highlightZonesGpu : fc.highlightZonesCpu;
        for( auto& e : fc.results )
        {
            if( llabs( e.timeA - e.timeB ) < thr ) continue;
            fc.highlightSrcLoc.emplace( e.srcloc );
            auto pit = inRangePtrs.find( e.srcloc );
            if( pit != inRangePtrs.end() )
            {
                for( auto ptr : pit->second ) zoneSet.emplace( ptr );
            }
        }
    }
}

void View::DrawFrameCompare()
{
    auto& fc = m_frameCompare;

    const auto scale = GetScale();
    ImGui::SetNextWindowSize( ImVec2( 700 * scale, 600 * scale ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "Frame compare", &fc.show, ImGuiWindowFlags_NoScrollbar );
    if( ImGui::GetCurrentWindowRead()->SkipItems ) { ImGui::End(); return; }

    if( !m_worker.AreSourceLocationZonesReady() && !m_worker.AreGpuSourceLocationZonesReady() )
    {
        ImGui::TextWrapped( "Zone statistics are still being computed. Please wait..." );
        ImGui::End();
        return;
    }

    const FrameData* fd = fc.frameSet ? fc.frameSet : m_worker.GetFramesBase();

    ImGui::TextWrapped( "Click two frames on the frame strip to compare them zone by zone. The first click picks frame A, the second picks frame B; the next click starts a new pair. Zones whose timing differs the most between the two frames are listed below and highlighted on the timeline (only the instances inside the two picked frames)." );
    const char* nextPick = ( fc.frameA < 0 || fc.frameB >= 0 ) ? "frame A" : "frame B";
    ImGui::TextDisabled( ICON_FA_HAND_POINTER " Next click on the frame strip selects: " );
    ImGui::SameLine();
    TextColoredUnformatted( 0xFF00A5FF, nextPick );
    ImGui::SameLine();
    if( ImGui::SmallButton( "Clear selection" ) )
    {
        fc.frameA = fc.frameB = -1;
        fc.MarkDirty();
    }
    ImGui::Separator();

    // Frame set selector (in case of multiple frame sets).
    const auto& frameSets = m_worker.GetFrames();
    if( frameSets.size() > 1 )
    {
        if( ImGui::BeginCombo( "Frame set", fd ? GetFrameSetName( *fd ) : "" ) )
        {
            for( auto& f : frameSets )
            {
                if( ImGui::Selectable( GetFrameSetName( *f ), f == fd ) )
                {
                    fc.frameSet = f;
                    fc.Reset();
                }
            }
            ImGui::EndCombo();
        }
    }

    const int frameCnt = fd ? (int)m_worker.GetFrameCount( *fd ) : 0;

    // Frame A / B inputs.
    bool changed = false;
    ImGui::PushItemWidth( 120 * scale );
    int fa = fc.frameA;
    if( ImGui::InputInt( "Frame A", &fa ) )
    {
        fc.frameSet = fd;
        fc.frameA = std::max( -1, std::min( frameCnt - 1, fa ) );
        changed = true;
    }
    ImGui::SameLine();
    if( fc.frameA >= 0 && fc.frameA < frameCnt )
    {
        ImGui::TextDisabled( "%s", TimeToString( m_worker.GetFrameTime( *fd, fc.frameA ) ) );
    }
    else
    {
        ImGui::TextDisabled( "(not set)" );
    }

    int fbi = fc.frameB;
    if( ImGui::InputInt( "Frame B", &fbi ) )
    {
        fc.frameSet = fd;
        fc.frameB = std::max( -1, std::min( frameCnt - 1, fbi ) );
        changed = true;
    }
    ImGui::SameLine();
    if( fc.frameB >= 0 && fc.frameB < frameCnt )
    {
        ImGui::TextDisabled( "%s", TimeToString( m_worker.GetFrameTime( *fd, fc.frameB ) ) );
    }
    else
    {
        ImGui::TextDisabled( "(not set)" );
    }
    ImGui::PopItemWidth();

    ImGui::Separator();

    // Zone domain selector. GPU is default because GPU-bound variation is the common case here.
    int mode = fc.mode;
    const bool hasGpu = m_worker.AreGpuSourceLocationZonesReady() && !m_worker.GetGpuSourceLocationZones().empty();
    const bool hasCpu = m_worker.AreSourceLocationZonesReady();
    if( ImGui::RadioButton( "GPU zones", &mode, 0 ) && mode != fc.mode ) { fc.mode = 0; changed = true; }
    ImGui::SameLine();
    if( ImGui::RadioButton( "CPU zones", &mode, 1 ) && mode != fc.mode ) { fc.mode = 1; changed = true; }
    if( fc.mode == 0 && !hasGpu )
    {
        ImGui::SameLine();
        TextColoredUnformatted( 0xFF4444FF, ICON_FA_TRIANGLE_EXCLAMATION " No GPU zones in this trace" );
    }
    if( fc.mode == 1 && !hasCpu )
    {
        ImGui::SameLine();
        TextColoredUnformatted( 0xFF4444FF, ICON_FA_TRIANGLE_EXCLAMATION " No CPU zones in this trace" );
    }

    if( fc.mode == 1 )
    {
        int acc = (int)fc.accumulationMode;
        ImGui::TextUnformatted( "Timing" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( ImGui::CalcTextSize( "Non-reentrant" ).x + ImGui::GetTextLineHeight() * 2 );
        if( ImGui::Combo( "##acc", &acc, "Self only\0With children\0Non-reentrant\0" ) )
        {
            fc.accumulationMode = (AccumulationMode)acc;
            changed = true;
        }
    }

    ImGui::Checkbox( "Highlight on timeline", &fc.highlightDiff );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 160 * scale );
    if( ImGui::SliderFloat( "Threshold", &fc.highlightThreshold, 0.f, 50.f, "%.1f%% of frame delta" ) )
    {
        changed = true;
    }
    DrawHelpMarker( "A zone is highlighted when its time difference between the two frames is at least this fraction of the overall frame-time difference." );

    if( changed ) fc.MarkDirty();

    const bool ready = fd && fc.frameA >= 0 && fc.frameA < frameCnt && fc.frameB >= 0 && fc.frameB < frameCnt;
    if( !ready )
    {
        ImGui::Separator();
        ImGui::PushFont( g_fonts.normal, FontBig );
        ImGui::Dummy( ImVec2( 0, ( ImGui::GetContentRegionAvail().y - ImGui::GetTextLineHeight() * 2 ) * 0.5f ) );
        TextCentered( ICON_FA_LEFT_RIGHT );
        TextCentered( "Select frame A and frame B" );
        ImGui::PopFont();
        ImGui::End();
        return;
    }

    if( fc.dirty ) ComputeFrameCompare();

    ImGui::Separator();
    TextFocused( "Frame A time:", TimeToString( fc.frameTimeA ) );
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    TextFocused( "Frame B time:", TimeToString( fc.frameTimeB ) );
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    const auto frameDelta = fc.frameTimeA - fc.frameTimeB;
    TextFocused( "Delta:", TimeToString( frameDelta ) );

    if( fc.results.empty() )
    {
        ImGui::TextDisabled( "No zones found inside the selected frames." );
        ImGui::End();
        return;
    }

    ImGui::BeginChild( "##framecompare" );
    if( ImGui::BeginTable( "##framecomparetable", 6,
        ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY ) )
    {
        ImGui::TableSetupScrollFreeze( 0, 1 );
        ImGui::TableSetupColumn( "Name", ImGuiTableColumnFlags_NoHide );
        ImGui::TableSetupColumn( "Location" );
        ImGui::TableSetupColumn( "Frame A", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize );
        ImGui::TableSetupColumn( "Frame B", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize );
        ImGui::TableSetupColumn( "Delta", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize );
        ImGui::TableSetupColumn( "% of frame delta", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoResize );
        ImGui::TableHeadersRow();

        auto& highlightSet = fc.highlightSrcLoc;
        const auto refDelta = frameDelta != 0 ? llabs( frameDelta ) : std::max( fc.frameTimeA, fc.frameTimeB );

        for( auto& v : fc.results )
        {
            const auto delta = v.timeA - v.timeB;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();

            ImGui::PushID( v.srcloc );
            auto& srcloc = m_worker.GetSourceLocation( v.srcloc );
            auto name = m_worker.GetString( srcloc.name.active ? srcloc.name : srcloc.function );
            SmallColorBox( GetSrcLocColor( srcloc, 0 ) );
            ImGui::SameLine();
            const bool highlighted = highlightSet.find( v.srcloc ) != highlightSet.end();
            if( ImGui::Selectable( name, highlighted, ImGuiSelectableFlags_SpanAllColumns ) )
            {
                m_zoneSrcLocHighlight = v.srcloc;
                if( !fc.resultsAreGpu ) m_findZone.ShowZone( v.srcloc, name );
            }

            ImGui::TableNextColumn();
            const auto file = m_worker.GetString( srcloc.file );
            TextDisabledUnformatted( LocationToString( file, srcloc.line ) );
            if( ImGui::IsItemHovered() )
            {
                DrawSourceTooltip( file, srcloc.line );
                if( ImGui::IsItemClicked( 1 ) )
                {
                    if( SourceFileValid( file, m_worker.GetCaptureTime(), *this, m_worker ) )
                    {
                        ViewSourceCheckKeyMod( file, srcloc.line, m_worker.GetString( srcloc.function ) );
                    }
                }
            }

            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( v.timeA ) );
            if( v.cntA != 1 )
            {
                ImGui::SameLine();
                ImGui::TextDisabled( "(%s)", RealToString( v.cntA ) );
            }

            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( v.timeB ) );
            if( v.cntB != 1 )
            {
                ImGui::SameLine();
                ImGui::TextDisabled( "(%s)", RealToString( v.cntB ) );
            }

            ImGui::TableNextColumn();
            const uint32_t deltaColor = delta > 0 ? 0xFF6666EE : ( delta < 0 ? 0xFF66EE66 : 0xFFAAAAAA );
            TextColoredUnformatted( deltaColor, TimeToString( delta ) );

            ImGui::TableNextColumn();
            if( refDelta != 0 )
            {
                char buf[64];
                PrintStringPercent( buf, 100.0 * delta / refDelta );
                TextColoredUnformatted( deltaColor, buf );
            }
            else
            {
                ImGui::TextDisabled( "--" );
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();

    ImGui::End();
}

#else

void View::ComputeFrameCompare() {}

void View::DrawFrameCompare()
{
    auto& fc = m_frameCompare;
    ImGui::Begin( "Frame compare", &fc.show );
    ImGui::TextWrapped( "Frame compare requires statistics, which are disabled in this build (TRACY_NO_STATISTICS)." );
    ImGui::End();
}

#endif

}
