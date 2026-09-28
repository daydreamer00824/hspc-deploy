#ifndef CKDTREE_CPP_DECL
#define CKDTREE_CPP_DECL

/*
 * 取自 SciPy 1.15.2 scipy/spatial/ckdtree/src/ckdtree_decl.h（BSD-3-Clause，见 LICENSE.scipy.txt）。
 * 唯一改动：去掉对 numpy/npy_common.h 的依赖，LIKELY/UNLIKELY 宏和 npy_intp 换成等价的 GCC 内建 / intptr_t，
 * CKDTREE_PREFETCH 改为空操作（预取只影响速度），并去掉本项目用不到的函数声明（只保留建树与 k 近邻查询）。同目录下的其余文件（build.cxx、query.cxx、
 * distance*.h、rectangle.h 等）与 SciPy 原文件逐字节相同。
 * 之所以直接带上这份代码而不用 nanoflann 之类的库：kNN 在距离并列时返回哪个点取决于建树（std::partition）
 * 与查询（优先队列）的具体实现，而 PC 编码器的输入正是这 15 个近邻的顺序；用同一份代码才能保证并列时的取舍与
 * Python 参考实现（scipy.spatial.cKDTree）完全一致。
 */
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
#define CKDTREE_LIKELY(x) __builtin_expect(!!(x), 1)
#define CKDTREE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define CKDTREE_PREFETCH(x, rw, loc) ((void)0)  /* 预取只影响速度，不影响结果 */

#define ckdtree_intp_t intptr_t
#define ckdtree_fmin(x, y)   fmin(x, y)
#define ckdtree_fmax(x, y)   fmax(x, y)
#define ckdtree_fabs(x)   fabs(x)

#include "ordered_pair.h"
#include "coo_entries.h"

struct ckdtreenode {
    ckdtree_intp_t      split_dim;
    ckdtree_intp_t      children;
    double   split;
    ckdtree_intp_t      start_idx;
    ckdtree_intp_t      end_idx;
    ckdtreenode   *less;
    ckdtreenode   *greater;
    ckdtree_intp_t      _less;
    ckdtree_intp_t      _greater;
};

struct ckdtree {
    // tree structure
    std::vector<ckdtreenode>  *tree_buffer;
    ckdtreenode   *ctree;
    // meta data
    double   *raw_data;
    ckdtree_intp_t      n;
    ckdtree_intp_t      m;
    ckdtree_intp_t      leafsize;
    double   *raw_maxes;
    double   *raw_mins;
    ckdtree_intp_t      *raw_indices;
    double   *raw_boxsize_data;
    ckdtree_intp_t size;
};

int
build_ckdtree(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
              double *maxes, double *mins, int _median, int _compact);

int
query_knn(const ckdtree     *self,
          double       *dd,
          ckdtree_intp_t          *ii,
          const double *xx,
          const ckdtree_intp_t     n,
          const ckdtree_intp_t     *k,
          const ckdtree_intp_t     nk,
          const ckdtree_intp_t     kmax,
          const double  eps,
          const double  p,
          const double  distance_upper_bound);

#endif
