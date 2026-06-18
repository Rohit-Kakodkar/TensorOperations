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
  using result_type = interm_type;

  node_type   node;
  tiling_type tiling;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t) : node(n), tiling(t) {}

  KOKKOS_FUNCTION result_type
  operator()(const typename Kokkos::TeamPolicy<exec_space>::member_type& team,
             Kokkos::Array<int, Rank> tile_idx) const;
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
