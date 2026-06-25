#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "TracyFilesystem.hpp"
#include "TracyImGui.hpp"
#include "TracyPrint.hpp"
#include "TracyView.hpp"
#include "tracy_pdqsort.h"
#include "../Fonts.hpp"

namespace tracy
{

#ifndef TRACY_NO_STATISTICS

// Iterate the direct children of a zone, handling both storage forms
// (inline "magic" Vector<T> by value, or Vector<short_ptr<T>>).
template<typename T, typename F>
static void ForEachChild( const Vector<short_ptr<T>>& ch, F&& f )
{
    if( ch.is_magic() )
    {
        auto& vec = *(Vector<T>*)&ch;
        for( auto& c : vec ) f( c );
    }
    else
    {
        for( auto& cp : ch ) f( *cp );
    }
}

void View::ShowChildStats( int16_t srcloc, bool gpu )
{
    auto& cs = m_childStats;
    cs.show = true;
    cs.srcloc = srcloc;
    cs.gpu = gpu;
    // Default the range: follow the Statistics range if set, else the current view.
    cs.range.active = true;
    if( m_statRange.active )
    {
        cs.range.min = m_statRange.min;
        cs.range.max = m_statRange.max;
    }
    else
    {
        cs.range.min = m_vd.zvStart;
        cs.range.max = m_vd.zvEnd;
    }
}

void View::DrawChildStats()
{
    auto& cs = m_childStats;
    const auto scale = GetScale();
    ImGui::SetNextWindowSize( ImVec2( 720 * scale, 600 * scale ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "Child zones over range", &cs.show, ImGuiWindowFlags_NoScrollbar );
    if( ImGui::GetCurrentWindowRead()->SkipItems ) { ImGui::End(); return; }
    // Note: srcloc may be negative — runtime-allocated source locations (dynamic
    // zone names like "mesh_commands_total#526") use negative ids, and both
    // GetSourceLocation and the zone maps handle them. The window only draws
    // after ShowChildStats set a real srcloc, so no "is set" check is needed.

    auto& psl = m_worker.GetSourceLocation( cs.srcloc );
    SmallColorBox( GetSrcLocColor( psl, 0 ) );
    ImGui::SameLine();
    ImGui::TextUnformatted( m_worker.GetZoneName( psl ) );
    ImGui::SameLine();
    ImGui::TextDisabled( cs.gpu ? "(GPU)" : "(CPU)" );
    TextDisabledUnformatted( LocationToString( m_worker.GetString( psl.file ), psl.line ) );

    ImGui::Separator();

    // Range controls.
    auto& r = cs.range;
    ImGui::Checkbox( "Limit to range", &r.active );
    if( r.active )
    {
        ImGui::SameLine();
        if( ImGui::SmallButton( ICON_FA_MAGNIFYING_GLASS " From view" ) ) { r.min = m_vd.zvStart; r.max = m_vd.zvEnd; }
        if( m_statRange.active )
        {
            ImGui::SameLine();
            if( ImGui::SmallButton( ICON_FA_ARROW_UP_WIDE_SHORT " From stat range" ) ) { r.min = m_statRange.min; r.max = m_statRange.max; }
        }
        if( r.min > r.max ) std::swap( r.min, r.max );
        TextFocused( "Range:", TimeToString( r.max - r.min ) );
        ImGui::SameLine();
        ImGui::TextDisabled( "%s \xe2\x80\x93 %s", TimeToStringExact( r.min ), TimeToStringExact( r.max ) );
    }
    else
    {
        ImGui::SameLine();
        ImGui::TextDisabled( "(whole trace)" );
    }

    const auto rmin = r.active ? r.min : std::numeric_limits<int64_t>::min();
    const auto rmax = r.active ? r.max : std::numeric_limits<int64_t>::max();

    // Aggregate: for every instance of this source location whose start is in range,
    // group its direct children by child source location.
    struct ChildAgg
    {
        int16_t srcloc;
        int64_t total = 0;
        int64_t mn = std::numeric_limits<int64_t>::max();
        int64_t mx = 0;
        double sumSq = 0;
        std::vector<int64_t> durs;
    };
    unordered_flat_map<int16_t, ChildAgg> cmap;
    int64_t parentTotal = 0, childrenTotal = 0;
    size_t parentCount = 0;

    auto addChild = [&]( int16_t sl, int64_t dur )
    {
        auto it = cmap.find( sl );
        if( it == cmap.end() ) it = cmap.emplace( sl, ChildAgg { sl } ).first;
        auto& a = it->second;
        a.total += dur;
        a.mn = std::min( a.mn, dur );
        a.mx = std::max( a.mx, dur );
        a.sumSq += (double)dur * (double)dur;
        a.durs.push_back( dur );
        childrenTotal += dur;
    };

    if( !cs.gpu )
    {
        if( m_worker.AreSourceLocationZonesReady() )
        {
            auto& slz = m_worker.GetZonesForSourceLocation( cs.srcloc );
            for( auto& zt : slz.zones )
            {
                auto& z = *zt.Zone();
                const auto start = z.Start();
                if( start < rmin || start >= rmax ) continue;
                parentTotal += m_worker.GetZoneEnd( z ) - start;
                parentCount++;
                if( !z.HasChildren() ) continue;
                ForEachChild<ZoneEvent>( m_worker.GetZoneChildren( z.Child() ), [&]( const ZoneEvent& c ) {
                    addChild( c.SrcLoc(), m_worker.GetZoneEnd( c ) - c.Start() );
                } );
            }
        }
    }
    else
    {
        if( m_worker.AreGpuSourceLocationZonesReady() )
        {
            auto& gslz = m_worker.GetGpuSourceLocationZones();
            auto sit = gslz.find( cs.srcloc );
            if( sit != gslz.end() )
            {
                for( auto& zt : sit->second.zones )
                {
                    auto& z = *zt.Zone();
                    const auto start = z.GpuStart();
                    if( start < 0 || start < rmin || start >= rmax ) continue;
                    parentTotal += m_worker.GetZoneEnd( z ) - start;
                    parentCount++;
                    if( z.Child() < 0 ) continue;
                    ForEachChild<GpuEvent>( m_worker.GetGpuChildren( z.Child() ), [&]( const GpuEvent& c ) {
                        addChild( c.SrcLoc(), m_worker.GetZoneEnd( c ) - c.GpuStart() );
                    } );
                }
            }
        }
    }

    ImGui::Separator();
    TextFocused( "Zone instances:", RealToString( parentCount ) );
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    TextFocused( "Total time:", TimeToString( parentTotal ) );
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    const auto selfTotal = parentTotal - childrenTotal;
    TextFocused( "Self time:", TimeToString( selfTotal ) );
    if( parentTotal > 0 )
    {
        ImGui::SameLine();
        char sb[64];
        PrintStringPercent( sb, 100. * selfTotal / parentTotal );
        TextDisabledUnformatted( sb );
    }

    if( cmap.empty() )
    {
        ImGui::TextDisabled( "No child zones for this selection." );
        ImGui::End();
        return;
    }

    std::vector<ChildAgg*> rows;
    rows.reserve( cmap.size() );
    for( auto& it : cmap )
    {
        std::sort( it.second.durs.begin(), it.second.durs.end() );
        rows.push_back( &it.second );
    }

    ImGui::BeginChild( "##childstats" );
    if( ImGui::BeginTable( "##cst", 9, ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY ) )
    {
        ImGui::TableSetupScrollFreeze( 0, 1 );
        ImGui::TableSetupColumn( "Child zone", ImGuiTableColumnFlags_NoHide );
        ImGui::TableSetupColumn( "Location" );
        ImGui::TableSetupColumn( "Count", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableSetupColumn( "Total", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableSetupColumn( "Mean", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableSetupColumn( ICON_FA_WAVE_SQUARE " Std (\xcf\x83)", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableSetupColumn( "Median", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableSetupColumn( "Min", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableSetupColumn( "Max", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed );
        ImGui::TableHeadersRow();

        auto median = []( const ChildAgg* a ) { return a->durs[a->durs.size() / 2]; };
        auto stddev = []( const ChildAgg* a ) -> double {
            const double n = (double)a->durs.size();
            const double mean = (double)a->total / n;
            const double var = a->sumSq / n - mean * mean;
            return var > 0 ? std::sqrt( var ) : 0.0;
        };
        const auto& sortspec = *ImGui::TableGetSortSpecs()->Specs;
        const bool asc = sortspec.SortDirection == ImGuiSortDirection_Ascending;
        auto cmp = [&]( const ChildAgg* a, const ChildAgg* b ) -> bool {
            double x = 0, y = 0;
            switch( sortspec.ColumnIndex )
            {
            case 0: { int c = strcmp( m_worker.GetZoneName( m_worker.GetSourceLocation( a->srcloc ) ), m_worker.GetZoneName( m_worker.GetSourceLocation( b->srcloc ) ) ); return asc ? c < 0 : c > 0; }
            case 2: x = (double)a->durs.size(); y = (double)b->durs.size(); break;
            case 4: x = (double)a->total / a->durs.size(); y = (double)b->total / b->durs.size(); break;
            case 5: x = stddev( a ); y = stddev( b ); break;
            case 6: x = (double)median( a ); y = (double)median( b ); break;
            case 7: x = (double)a->mn; y = (double)b->mn; break;
            case 8: x = (double)a->mx; y = (double)b->mx; break;
            default: x = (double)a->total; y = (double)b->total; break;  // column 1 (Location) falls back to total
            }
            return asc ? x < y : x > y;
        };
        pdqsort_branchless( rows.begin(), rows.end(), cmp );

        for( auto* a : rows )
        {
            const auto n = (int64_t)a->durs.size();
            auto& sl = m_worker.GetSourceLocation( a->srcloc );
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID( a->srcloc );
            SmallColorBox( GetSrcLocColor( sl, 0 ) );
            ImGui::SameLine();
            if( ImGui::Selectable( m_worker.GetZoneName( sl ), m_zoneSrcLocHighlight == a->srcloc, ImGuiSelectableFlags_SpanAllColumns ) )
            {
                m_zoneSrcLocHighlight = a->srcloc;
            }
            ImGui::TableNextColumn();
            const auto file = m_worker.GetString( sl.file );
            TextDisabledUnformatted( LocationToString( file, sl.line ) );
            if( ImGui::IsItemHovered() )
            {
                DrawSourceTooltip( file, sl.line );
                if( ImGui::IsItemClicked( 1 ) && SourceFileValid( file, m_worker.GetCaptureTime(), *this, m_worker ) )
                {
                    ViewSourceCheckKeyMod( file, sl.line, m_worker.GetString( sl.function ) );
                }
            }
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( RealToString( n ) );
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( a->total ) );
            if( parentTotal > 0 )
            {
                ImGui::SameLine();
                char b[64];
                PrintStringPercent( b, 100. * a->total / parentTotal );
                TextDisabledUnformatted( b );
            }
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( a->total / n ) );
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( (int64_t)stddev( a ) ) );
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( median( a ) ) );
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( a->mn ) );
            ImGui::TableNextColumn();
            ImGui::TextUnformatted( TimeToString( a->mx ) );
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();

    ImGui::End();
}

#else

void View::ShowChildStats( int16_t, bool ) {}
void View::DrawChildStats() {}

#endif

}
