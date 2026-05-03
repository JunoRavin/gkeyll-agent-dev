/* -*- c -*- */

#include <stdio.h>

extern "C" {
#include <gkyl_rect_grid.h>
#include <gkyl_util.h>
#include <gkyl_alloc.h>
  int cu_rect_grid_test(const struct gkyl_rect_grid grid);
}

__global__
void ker_cu_rect_grid_test(const struct gkyl_rect_grid grid, int *nfail)
{
  *nfail = 0;

  double lower[] = {1.0, 1.0}, upper[] = {2.5, 5.0};
  int cells[] = {20, 20};

  GKYL_CU_CHECK( grid.ndim == 2, nfail );
  for (int i=0; i<grid.ndim; ++i) {
    GKYL_CU_CHECK( grid.lower[i] == lower[i], nfail );
    GKYL_CU_CHECK( grid.upper[i] == upper[i], nfail );
    GKYL_CU_CHECK( grid.cells[i] == cells[i], nfail );
    // Tolerance comparison instead of bit-exact equality: under hipcc with
    // -ffast-math the device path can lower (a-b)/c into (a-b)*v_rcp_f64(c),
    // and v_rcp_f64 on AMDGPU is not IEEE-strict — a few ULPs off the host
    // division. The test's intent is "kernel sees the right dx", not
    // "host and device division agree bit-for-bit".
    double expected_dx = (upper[i]-lower[i])/cells[i];
    GKYL_CU_CHECK( fabs(grid.dx[i] - expected_dx) <= 1e-14, nfail );
  }
  GKYL_CU_CHECK( grid.cellVolume == 0.075*0.2, nfail );
  
}

int cu_rect_grid_test(const struct gkyl_rect_grid grid)
{
  int *nfail_dev = (int *) gkyl_cu_malloc(sizeof(int));  
  ker_cu_rect_grid_test<<<1,1>>>(grid, nfail_dev);

  int nfail;
  gkyl_cu_memcpy(&nfail, nfail_dev, sizeof(int), GKYL_CU_MEMCPY_D2H);
  gkyl_cu_free(nfail_dev);

  return nfail;  
}
