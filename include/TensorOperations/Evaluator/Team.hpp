#pragma once
// Included from within TensorOperations namespace by Evaluator.hpp

// ---------------------------------------------------------------------------
// Specialization 2: TeamPolicyTag + InputTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
template <TensorLike T, typename HookOp, typename Tile_>
struct Evaluator<TeamPolicyTag, NodeHandle<InputTag, T, HookOp>, Tile_> {
  using node_type           = NodeHandle<InputTag, T, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag;
  static constexpr int Rank = tiling_type::rank;
  using value_type          = typename node_type::value_type;
  using exec_space          = typename Impl::exec_space_of<T>::type;
  using scratch_space       = typename exec_space::scratch_memory_space;
  using scratch_view_t =
      typename Impl::KokkosViewN<value_type, Rank, scratch_space>::type;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type   = interm_type;
  using team_member_t = typename Kokkos::TeamPolicy<exec_space>::member_type;
  using tiled_input_t = TiledView<TensorHandle<T>, tiling_type>;

  static_assert(node_type::Rank == Rank,
                "input staging tile must carry one extent per input mode");

  node_type     node;
  tiling_type   tiling;
  tiled_input_t tiled_input_;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t)
      : node(n), tiling(t), tiled_input_(tile_view(n.handle, t)) {}

  KOKKOS_FUNCTION result_type operator()(
      const team_member_t& team, Kokkos::Array<int, Rank> tile_idx) const {
    TIMING_SCOPE_ENTER(g_timing_stats.scratch_input_load_time,
                       g_timing_stats.scratch_input_load_count);
    result_type out{};
    for (int d = 0; d < Rank; ++d) out.shape_[d] = tiling.extent(d);
    out.modes_   = node.modes();
    out.storage_ = alloc_scratch(team, std::make_index_sequence<Rank>{});
    fill_team(team, tile_idx, out.storage_);
    TIMING_SCOPE_EXIT(g_timing_stats.scratch_input_load_time,
                      g_timing_stats.scratch_input_load_count);
    return out;
  }

 private:
  template <std::size_t... Is>
  KOKKOS_FUNCTION scratch_view_t
  alloc_scratch(const team_member_t& team, std::index_sequence<Is...>) const {
    return scratch_view_t(team.team_scratch(0),
                          static_cast<std::size_t>(tiling.extent(Is))...);
  }

  KOKKOS_FUNCTION void fill_team(const team_member_t&            team,
                                 const Kokkos::Array<int, Rank>& tile_idx,
                                 scratch_view_t&                 result) const {
    const auto sv        = subview_tile(tiled_input_, tile_idx);
    const auto sv_layout = sv.layout();

    const auto total = sv.size();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, total), [=](int i) {
      const auto coord = sv_layout[i];
      result[coord]    = sv[coord];
    });
  }
};

// ---------------------------------------------------------------------------
// Specialization 4: TeamPolicyTag + ContractionTag + Tile_  (scratch tier)
// ---------------------------------------------------------------------------
template <typename NA, typename NB, typename IntCRank, typename S, typename ES,
          typename HookOp, typename Tile_>
struct Evaluator<TeamPolicyTag,
                 NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>,
                 Tile_> {
  using node_type = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp>;
  using tiling_type         = Tile_;
  using policy_tag          = TeamPolicyTag;
  static constexpr int Rank = node_type::Rank;
  using value_type          = S;
  using exec_space          = ES;
  using scratch_space       = typename ES::scratch_memory_space;
  using scratch_view_t =
      typename Impl::KokkosViewN<value_type, Rank, scratch_space>::type;
  using interm_type =
      NodeHandle<IntermTag, scratch_view_t, std::integral_constant<int, Rank>,
                 exec_space, NoHook>;
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t) : node(n), tiling(t) {}

  KOKKOS_FUNCTION result_type
  operator()(const typename Kokkos::TeamPolicy<exec_space>::member_type& team,
             Kokkos::Array<int, Rank> tile_idx) const;
};
