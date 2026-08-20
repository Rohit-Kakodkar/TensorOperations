#pragma once

#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>

#include <cstddef>
#include <set>
#include <vector>

namespace sfpp_min {

struct MeshDims {
  int nex = 0;
  int ney = 0;
  int nez = 0;

  int         nspec_grid() const { return nex * ney * nez; }
  int         gnx() const { return DEG * nex + 1; }
  int         gny() const { return DEG * ney + 1; }
  int         gnz() const { return DEG * nez + 1; }
  std::size_t nnode_grid() const {
    return static_cast<std::size_t>(gnx()) * gny() * gnz();
  }
};

inline int grid_element_index(const MeshDims& d, int ex, int ey, int ez) {
  return ex + d.nex * (ey + d.ney * ez);
}

inline void grid_element_coords(const MeshDims& d, int igrid, int& ex, int& ey,
                                int& ez) {
  ex = igrid % d.nex;
  ey = (igrid / d.nex) % d.ney;
  ez = igrid / (d.nex * d.ney);
}

inline std::size_t topological_id(const MeshDims& d, int igrid, int iz, int iy,
                                  int ix) {
  int ex, ey, ez;
  grid_element_coords(d, igrid, ex, ey, ez);
  const std::size_t gx = static_cast<std::size_t>(DEG) * ex + ix;
  const std::size_t gy = static_cast<std::size_t>(DEG) * ey + iy;
  const std::size_t gz = static_cast<std::size_t>(DEG) * ez + iz;
  return gx + static_cast<std::size_t>(d.gnx()) * (gy + d.gny() * gz);
}

struct ElementSet {
  std::vector<int> to_grid;
  int              nspec() const { return static_cast<int>(to_grid.size()); }
};

inline ElementSet all_elements(const MeshDims& d) {
  ElementSet s;
  s.to_grid.reserve(d.nspec_grid());
  for (int i = 0; i < d.nspec_grid(); ++i) s.to_grid.push_back(i);
  return s;
}

inline ElementSet interior_elements(const MeshDims& d) {
  ElementSet s;
  for (int ez = 0; ez < d.nez; ++ez)
    for (int ey = 0; ey < d.ney; ++ey)
      for (int ex = 0; ex < d.nex; ++ex) {
        const bool shell = (ex == 0) || (ex == d.nex - 1) || (ey == 0) ||
                           (ey == d.ney - 1) || (ez == 0);
        if (!shell) s.to_grid.push_back(grid_element_index(d, ex, ey, ez));
      }
  return s;
}

inline int renumber_access_order(const MeshDims& d, const ElementSet& set,
                                 IglobMap& out) {
  const int        nspec = set.nspec();
  std::vector<int> dedup(d.nnode_grid(), -1);
  auto&            h     = out.h_map;
  int              count = 0;
  for (int ichunk = 0; ichunk < nspec; ichunk += kStorageChunk) {
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          for (int ielement = 0; ielement < kStorageChunk; ++ielement) {
            const int ispec = ichunk + ielement;
            if (ispec >= nspec) break;
            const std::size_t old =
                topological_id(d, set.to_grid[ispec], iz, iy, ix);
            if (dedup[old] == -1) dedup[old] = count++;
            h(ispec, iz, iy, ix) = dedup[old];
          }
        }
      }
    }
  }
  return count;
}

inline int renumber_grid_order(const MeshDims& d, const ElementSet& set,
                               IglobMap& out) {
  const int        nspec = set.nspec();
  std::vector<int> dedup(d.nnode_grid(), -1);
  auto&            h     = out.h_map;
  int              count = 0;
  for (int ispec = 0; ispec < nspec; ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const std::size_t old =
              topological_id(d, set.to_grid[ispec], iz, iy, ix);
          if (dedup[old] == -1) dedup[old] = count++;
          h(ispec, iz, iy, ix) = dedup[old];
        }
  return count;
}

inline double predicted_atomic_sectors_per_point(const ElementSet& set,
                                                 const IglobMap& g, int nglob) {
  const auto& h     = g.h_map;
  const int   nspec = set.nspec();
  const int   teams = num_teams(nspec);

  std::size_t           sectors = 0;
  std::size_t           points  = 0;
  std::set<std::size_t> touched;

  for (int team = 0; team < teams; ++team) {
    const int base = team * kExecChunk;
    for (int start = 0; start < kWorkItemsPerTeam; start += kWarpSize) {
      for (int comp = 0; comp < kComponents; ++comp) {
        touched.clear();
        for (int lane = 0; lane < kWarpSize; ++lane) {
          const int i = start + lane;
          if (i >= kWorkItemsPerTeam) break;
          const WorkItem w     = decompose(i);
          const int      ispec = base + w.ielement;
          if (ispec >= nspec) continue;
          const std::size_t iglob = h(ispec, w.iz, w.iy, w.ix);
          const std::size_t byte =
              sizeof(real_t) * (static_cast<std::size_t>(comp) * nglob + iglob);
          touched.insert(byte / 32u);
        }
        sectors += touched.size();
      }
      for (int lane = 0; lane < kWarpSize; ++lane) {
        const int i = start + lane;
        if (i >= kWorkItemsPerTeam) break;
        const WorkItem w = decompose(i);
        if (base + w.ielement < nspec) ++points;
      }
    }
  }
  return static_cast<double>(sectors) / static_cast<double>(points);
}

}  // namespace sfpp_min
