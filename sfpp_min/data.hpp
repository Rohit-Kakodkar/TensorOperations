#pragma once

#include <config.hpp>
#include <layout.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string>

namespace sfpp_min {

template <typename Offset>
struct DomainAccessor {
  Kokkos::View<real_t*> d;
  Offset                off;

  KOKKOS_INLINE_FUNCTION real_t& operator()(int ispec, int iz, int iy,
                                            int ix) const {
    return d(off(ispec, iz, iy, ix));
  }
};

template <typename Offset>
class DomainArray {
 public:
  using offset_type = Offset;
  using view_type   = Kokkos::View<real_t*>;
  using host_type   = typename view_type::host_mirror_type;

  DomainArray() = default;

  DomainArray(const std::string& label, int nspec)
      : nspec_(nspec),
        off_(nspec),
        d_(label, off_.span()),
        h_(Kokkos::create_mirror_view(d_)) {}

  real_t& host(int ispec, int iz, int iy, int ix) const {
    return h_(off_(ispec, iz, iy, ix));
  }

  DomainAccessor<Offset> accessor() const { return {d_, off_}; }

  void to_device() { Kokkos::deep_copy(d_, h_); }
  void to_host() { Kokkos::deep_copy(h_, d_); }

  int         nspec() const { return nspec_; }
  std::size_t span() const { return d_.extent(0); }
  std::size_t bytes() const { return span() * sizeof(real_t); }

  const view_type& device_view() const { return d_; }
  const host_type& host_view() const { return h_; }

 private:
  int       nspec_ = 0;
  Offset    off_;
  view_type d_;
  host_type h_;
};

template <typename Offset>
struct Metrics {
  DomainArray<Offset> xix, xiy, xiz;
  DomainArray<Offset> etax, etay, etaz;
  DomainArray<Offset> gammax, gammay, gammaz;
  DomainArray<Offset> jacobian;

  Metrics() = default;

  explicit Metrics(int nspec)
      : xix("xix", nspec),
        xiy("xiy", nspec),
        xiz("xiz", nspec),
        etax("etax", nspec),
        etay("etay", nspec),
        etaz("etaz", nspec),
        gammax("gammax", nspec),
        gammay("gammay", nspec),
        gammaz("gammaz", nspec),
        jacobian("jacobian", nspec) {}

  static constexpr int kArrays = 10;

  void to_device() {
    for_each([](DomainArray<Offset>& a) { a.to_device(); });
  }
  void to_host() {
    for_each([](DomainArray<Offset>& a) { a.to_host(); });
  }
  std::size_t bytes() const { return kArrays * xix.bytes(); }

  template <typename F>
  void for_each(F f) {
    f(xix);
    f(xiy);
    f(xiz);
    f(etax);
    f(etay);
    f(etaz);
    f(gammax);
    f(gammay);
    f(gammaz);
    f(jacobian);
  }
};

template <typename Offset>
struct Properties {
  DomainArray<Offset> kappa, mu, rho;

  Properties() = default;

  explicit Properties(int nspec)
      : kappa("kappa", nspec), mu("mu", nspec), rho("rho", nspec) {}

  static constexpr int kArrays = 3;

  void to_device() {
    kappa.to_device();
    mu.to_device();
    rho.to_device();
  }
  void to_host() {
    kappa.to_host();
    mu.to_host();
    rho.to_host();
  }
  std::size_t bytes() const { return kArrays * kappa.bytes(); }
};

inline constexpr int kComponents = 3;

struct Fields {
  using view_type = Kokkos::View<real_t**, Kokkos::LayoutLeft>;
  using host_type = typename view_type::host_mirror_type;

  view_type displacement, velocity, acceleration, mass_inverse;
  host_type h_displacement, h_velocity, h_acceleration, h_mass_inverse;

  Fields() = default;

  explicit Fields(int nglob)
      : displacement("displacement", nglob, kComponents),
        velocity("velocity", nglob, kComponents),
        acceleration("acceleration", nglob, kComponents),
        mass_inverse("mass_inverse", nglob, kComponents),
        h_displacement(Kokkos::create_mirror_view(displacement)),
        h_velocity(Kokkos::create_mirror_view(velocity)),
        h_acceleration(Kokkos::create_mirror_view(acceleration)),
        h_mass_inverse(Kokkos::create_mirror_view(mass_inverse)) {}

  void to_device() {
    Kokkos::deep_copy(displacement, h_displacement);
    Kokkos::deep_copy(velocity, h_velocity);
    Kokkos::deep_copy(acceleration, h_acceleration);
    Kokkos::deep_copy(mass_inverse, h_mass_inverse);
  }
  void to_host() {
    Kokkos::deep_copy(h_displacement, displacement);
    Kokkos::deep_copy(h_velocity, velocity);
    Kokkos::deep_copy(h_acceleration, acceleration);
    Kokkos::deep_copy(h_mass_inverse, mass_inverse);
  }

  int         nglob() const { return static_cast<int>(displacement.extent(0)); }
  std::size_t bytes() const {
    return 4u * displacement.size() * sizeof(real_t);
  }
};

struct IglobMap {
  using view_type = Kokkos::View<int****, Kokkos::LayoutLeft>;
  using host_type = typename view_type::host_mirror_type;

  view_type map;
  host_type h_map;

  IglobMap() = default;

  explicit IglobMap(int nspec)
      : map("iglob", nspec, NGLL, NGLL, NGLL),
        h_map(Kokkos::create_mirror_view(map)) {}

  void to_device() { Kokkos::deep_copy(map, h_map); }
  void to_host() { Kokkos::deep_copy(h_map, map); }

  int         nspec() const { return static_cast<int>(map.extent(0)); }
  std::size_t bytes() const { return map.size() * sizeof(int); }
};

}  // namespace sfpp_min
