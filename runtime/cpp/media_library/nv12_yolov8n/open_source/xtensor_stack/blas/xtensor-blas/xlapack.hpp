/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XLAPACK_HPP
#define XLAPACK_HPP

#include <algorithm>

#include "xtl/xcomplex.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xstorage.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xutils.hpp"

#include "xflens/cxxlapack/cxxlapack.cxx"

#include "xtensor-blas/xblas_config.hpp"
#include "xtensor-blas/xblas_utils.hpp"

namespace xt
{

namespace lapack
{
    /**
     * Interface to LAPACK gesv.
     */
    template <class E, class F>
    int gesv(E& A, F& b)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);
        XTENSOR_ASSERT(b.dimension() <= 2);
        XTENSOR_ASSERT(b.layout() == layout_type::column_major);

        uvector<blas_index_t> piv(A.shape()[0]);

        blas_index_t b_dim = b.dimension() > 1 ? static_cast<blas_index_t>(b.shape().back()) : 1;
        blas_index_t b_stride = b_dim == 1 ? static_cast<blas_index_t>(b.shape().front()) : stride_back(b);

        int info = cxxlapack::gesv<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            b_dim,
            A.data(),
            stride_back(A),
            piv.data(),
            b.data(),
            b_stride
        );

        return info;
    }

    template <class E, class F>
    auto getrf(E& A, F& piv)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        int info = cxxlapack::getrf<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            stride_back(A),
            piv.data()
        );

        return info;
    }

    template <class E, class T>
    inline auto orgqr(E& A, T& tau, blas_index_t n = -1)
    {
        using value_type = typename E::value_type;

        uvector<value_type> work(1);

        if (n == -1)
        {
            n = static_cast<blas_index_t>(A.shape()[1]);
        }

        blas_index_t m = static_cast<blas_index_t>(A.shape()[0]);
        blas_index_t a_stride = std::max(blas_index_t(1), m);

        int info = cxxlapack::orgqr<blas_index_t>(
            m,
            n,
            static_cast<blas_index_t>(tau.size()),
            A.data(),
            a_stride,
            tau.data(),
            work.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for orgqr.");
        }

        work.resize(static_cast<std::size_t>(work[0]));

        info = cxxlapack::orgqr<blas_index_t>(
            m,
            n,
            static_cast<blas_index_t>(tau.size()),
            A.data(),
            a_stride,
            tau.data(),
            work.data(),
            static_cast<blas_index_t>(work.size())
        );

        return info;
    }

    template <class E, class T>
    inline auto ungqr(E& A, T& tau, blas_index_t n = -1)
    {
        using value_type = typename E::value_type;

        uvector<value_type> work(1);

        if (n == -1)
        {
            n = static_cast<blas_index_t>(A.shape()[1]);
        }

        blas_index_t m = static_cast<blas_index_t>(A.shape()[0]);
        blas_index_t a_stride = std::max(blas_index_t(1), m);

        int info = cxxlapack::ungqr<blas_index_t>(
            m,
            n,
            static_cast<blas_index_t>(tau.size()),
            A.data(),
            a_stride,
            tau.data(),
            work.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for ungqr.");
        }

        work.resize(static_cast<std::size_t>(std::real(work[0])));

        info = cxxlapack::ungqr<blas_index_t>(
            m,
            n,
            static_cast<blas_index_t>(tau.size()),
            A.data(),
            a_stride,
            tau.data(),
            work.data(),
            static_cast<blas_index_t>(work.size())
        );

        return info;
    }

    template <class E, class T>
    int geqrf(E& A, T& tau)
    {
        using value_type = typename E::value_type;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        uvector<value_type> work(1);
        blas_index_t m = static_cast<blas_index_t>(A.shape()[0]);
        blas_index_t a_stride = std::max(blas_index_t(1), m);

        int info = cxxlapack::geqrf<blas_index_t>(
            m,
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            a_stride,
            tau.data(),
            work.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for geqrf.");
        }

        work.resize(static_cast<std::size_t>(std::real(work[0])));

        info = cxxlapack::geqrf<blas_index_t>(
            m,
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            a_stride,
            tau.data(),
            work.data(),
            static_cast<blas_index_t>(work.size())
        );

        return info;
    }

    namespace detail
    {
        template <class U, class VT>
        inline auto init_u_vt(U& u, VT& vt, char jobz, std::size_t m, std::size_t n)
        {
            // rules for sgesdd
            // u:
            //   if jobz == 'O' and M >= N, u is not referenced
            //   if jobz == 'N', u is also not referenced
            // vt:
            //   if jobz == 'O' and M < N vt is not referenced
            //   if jobz == 'N', vt is also not referenced
            if (jobz == 'A' || (jobz == 'O' && m < n))
            {
                u.resize({m, m});
            }
            if (jobz == 'A' || (jobz == 'O' && m >= n))
            {
                vt.resize({n, n});
            }
            if (jobz == 'S')
            {
                u.resize({m, std::min(m, n)});
                vt.resize({std::min(m, n), n});
            }
            if (jobz == 'N')
            {
                // u AND vt are unreferenced -- can't use strides().back()...
                return std::make_pair(1, 1);
            }
            if (jobz == 'O')
            {
                // u OR vt are unreferenced -- can't use strides().back()...
                return m >= n ? std::make_pair(1, stride_back(vt)) :
                                std::make_pair(stride_back(u), 1);
            }

            return std::make_pair(std::max(blas_index_t(u.shape()[0]), 1),
                                  std::max(blas_index_t(vt.shape()[0]), 1));
        }
    }

    template <class E, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto gesdd(E& A, char jobz = 'A')
    {
        using value_type = typename E::value_type;
        using xtype1 = xtensor<value_type, 1, layout_type::column_major>;
        using xtype2 = xtensor<value_type, 2, layout_type::column_major>;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        uvector<value_type> work(1);

        std::size_t m = A.shape()[0];
        std::size_t n = A.shape()[1];

        xtype1 s;
        s.resize({ std::max(static_cast<std::size_t>(1), std::min(m, n)) });

        xtype2 u, vt;

        blas_index_t u_stride, vt_stride;
        std::tie(u_stride, vt_stride) = detail::init_u_vt(u, vt, jobz, m, n);

        uvector<blas_index_t> iwork(8 * std::min(m, n));
        blas_index_t a_stride = static_cast<blas_index_t>(std::max(std::size_t(1), m));

        int info = cxxlapack::gesdd<blas_index_t>(
            jobz,
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            a_stride,
            s.data(),
            u.data(),
            u_stride,
            vt.data(),
            vt_stride,
            work.data(),
            static_cast<blas_index_t>(-1),
            iwork.data()
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for real gesdd.");
        }

        work.resize(static_cast<std::size_t>(work[0]));
        info = cxxlapack::gesdd<blas_index_t>(
            jobz,
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            a_stride,
            s.data(),
            u.data(),
            u_stride,
            vt.data(),
            vt_stride,
            work.data(),
            static_cast<blas_index_t>(work.size()),
            iwork.data()
        );

        return std::make_tuple(std::move(info), std::move(u), std::move(s), std::move(vt));
    }

    // Complex variant of gesdd
    template <class E, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    auto gesdd(E& A, char jobz = 'A')
    {
        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;
        using xtype1 = xtensor<underlying_value_type, 1, layout_type::column_major>;
        using xtype2 = xtensor<value_type, 2, layout_type::column_major>;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        std::size_t m = A.shape()[0];
        std::size_t n = A.shape()[1];

        uvector<value_type> work(1);
        uvector<underlying_value_type> rwork(1);
        uvector<blas_index_t> iwork(8 * std::min(m, n));

        std::size_t mx = std::max(m, n);
        std::size_t mn = std::min(m, n);
        if (jobz == 'N')
        {
            rwork.resize(5 * mn);
        }
        else if (mx > mn)
        {
            // TODO verify size
            rwork.resize(5 * mn * mn + 5 * mn);
        }
        else
        {
            // TODO verify size
            rwork.resize(std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn));
        }

        xtype1 s;
        s.resize({ std::max(static_cast<std::size_t>(1), std::min(m, n)) });

        xtype2 u, vt;

        blas_index_t u_stride, vt_stride;
        std::tie(u_stride, vt_stride) = detail::init_u_vt(u, vt, jobz, m, n);
        blas_index_t a_stride = static_cast<blas_index_t>(std::max(std::size_t(1), m));

        int info = cxxlapack::gesdd<blas_index_t>(
            jobz,
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            a_stride,
            s.data(),
            u.data(),
            u_stride,
            vt.data(),
            vt_stride,
            work.data(),
            static_cast<blas_index_t>(-1),
            rwork.data(),
            iwork.data()
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for complex gesdd.");
        }
        work.resize(static_cast<std::size_t>(std::real(work[0])));

        info = cxxlapack::gesdd<blas_index_t>(
            jobz,
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            A.data(),
            a_stride,
            s.data(),
            u.data(),
            u_stride,
            vt.data(),
            vt_stride,
            work.data(),
            static_cast<blas_index_t>(work.size()),
            rwork.data(),
            iwork.data()
        );

        return std::make_tuple(std::move(info), std::move(u), std::move(s), std::move(vt));
    }


    template <class E>
    int potr(E& A, char uplo = 'L')
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        int info = cxxlapack::potrf<blas_index_t>(
            uplo,
            static_cast<blas_index_t>(A.shape()[0]),
            A.data(),
            stride_back(A)
        );

        return info;
    }

    template <class E1, class E2>
    int potrs(E1& A, E2& b, char uplo = 'L')
    {
      XTENSOR_ASSERT(A.dimension() == 2);
      XTENSOR_ASSERT(A.layout() == layout_type::column_major);

      XTENSOR_ASSERT(b.dimension() == 1);

      XTENSOR_ASSERT(A.shape()[0] == A.shape()[1]);

      int info = cxxlapack::potrs<blas_index_t>(
          uplo,
          static_cast<blas_index_t>(A.shape()[0]),
          1,
          A.data(),
          static_cast<blas_index_t>(A.shape()[0]),
          b.data(),
          static_cast<blas_index_t>(b.shape()[0])
      );

      return info;
    }

    template <class E1, class E2>
    int trtrs(E1& A, E2& b, char uplo = 'L', char trans = 'N', char diag = 'N')
    {
      XTENSOR_ASSERT(A.dimension() == 2);
      XTENSOR_ASSERT(A.layout() == layout_type::column_major);

      XTENSOR_ASSERT(b.dimension() == 1);

      XTENSOR_ASSERT(A.shape()[0] == A.shape()[1]);

      int info = cxxlapack::trtrs<blas_index_t>(
          uplo,
          trans,
          diag,
          static_cast<blas_index_t>(A.shape()[0]),
          1,
          A.data(),
          static_cast<blas_index_t>(A.shape()[0]),
          b.data(),
          static_cast<blas_index_t>(b.shape()[0])
      );

      return info;
    }

    /**
     * Interface to LAPACK getri.
     *
     * @param A matrix to invert
     * @return inverse of A
     */
    template <class E>
    int getri(E& A, uvector<blas_index_t>& piv)
    {
        using value_type = typename E::value_type;

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        uvector<value_type> work(1);

        // get work size
        int info = cxxlapack::getri<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            A.data(),
            stride_back(A),
            piv.data(),
            work.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info > 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for getri.");
        }

        work.resize(static_cast<std::size_t>(std::real(work[0])));

        info = cxxlapack::getri<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            A.data(),
            stride_back(A),
            piv.data(),
            work.data(),
            static_cast<blas_index_t>(work.size())
        );

        return info;
    }

    /**
     * Interface to LAPACK geev.
     * @returns info
     */
    template <class E, class W, class V>
    int geev(E& A, char jobvl, char jobvr, W& wr, W& wi, V& VL, V& VR)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        using value_type = typename E::value_type;

        const auto N = A.shape()[0];
        uvector<value_type> work(1);

        int info = cxxlapack::geev<blas_index_t>(
            jobvl,
            jobvr,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            wr.data(),
            wi.data(),
            VL.data(),
            stride_back(VL),
            VR.data(),
            stride_back(VR),
            work.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for geev.");
        }

        work.resize(static_cast<std::size_t>(work[0]));

        info = cxxlapack::geev<blas_index_t>(
            jobvl,
            jobvr,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            wr.data(),
            wi.data(),
            VL.data(),
            stride_back(VL),
            VR.data(),
            stride_back(VR),
            work.data(),
            static_cast<blas_index_t>(work.size())
        );

        return info;
    }

    /**
     * Interface to LAPACK syevd.
     * @returns info
     */
    template <class E, class W>
    int syevd(E& A, char jobz, char uplo, W& w)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        using value_type = typename E::value_type;

        auto N = A.shape()[0];
        uvector<value_type> work(1);
        uvector<blas_index_t> iwork(1);

        int info = cxxlapack::syevd<blas_index_t>(
            jobz,
            uplo,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            w.data(),
            work.data(),
            static_cast<blas_index_t>(-1),
            iwork.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for syevd.");
        }

        work.resize(std::size_t(work[0]));
        iwork.resize(std::size_t(iwork[0]));

        info = cxxlapack::syevd<blas_index_t>(
            jobz,
            uplo,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            w.data(),
            work.data(),
            static_cast<blas_index_t>(work.size()),
            iwork.data(),
            static_cast<blas_index_t>(iwork.size())
        );

        return info;
    }

    /**
     * Interface to LAPACK sygvd.
     * @returns info
     */
    template <class E, class W>
    int sygvd(E& A, E& B, blas_index_t itype, char jobz, char uplo, W& w)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);
        XTENSOR_ASSERT(B.dimension() == 2);
        XTENSOR_ASSERT(B.layout() == layout_type::column_major);

        using value_type = typename E::value_type;

        auto N = A.shape()[0];
        XTENSOR_ASSERT(B.shape()[0] ==N);

        uvector<value_type> work(1);
        uvector<blas_index_t> iwork(1);

        int info = cxxlapack::sygvd<blas_index_t>(
            itype,
            jobz,
            uplo,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            B.data(),
            stride_back(B),
            w.data(),
            work.data(),
            static_cast<blas_index_t>(-1),
            iwork.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for sygvd.");
        }

        work.resize(std::size_t(work[0]));
        iwork.resize(std::size_t(iwork[0]));

        info = cxxlapack::sygvd<blas_index_t>(
            itype,
            jobz,
            uplo,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            B.data(),
            stride_back(B),
            w.data(),
            work.data(),
            static_cast<blas_index_t>(work.size()),
            iwork.data(),
            static_cast<blas_index_t>(iwork.size())
        );

        return info;
    }

    /**
     * Complex version of geev
     */
    template <class E, class W, class V>
    int geev(E& A, char jobvl, char jobvr, W& w, V& VL, V& VR)
    {
        // TODO implement for complex numbers

        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;

        const auto N = A.shape()[0];
        uvector<value_type> work(1);
        uvector<underlying_value_type> rwork(2 * N);

        int info = cxxlapack::geev<blas_index_t>(
            jobvl,
            jobvr,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            w.data(),
            VL.data(),
            stride_back(VL),
            VR.data(),
            stride_back(VR),
            work.data(),
            -1,
            rwork.data()
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for geev.");
        }

        work.resize(std::size_t(std::real(work[0])));

        info = cxxlapack::geev<blas_index_t>(
            jobvl,
            jobvr,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            w.data(),
            VL.data(),
            stride_back(VL),
            VR.data(),
            stride_back(VR),
            work.data(),
            static_cast<blas_index_t>(work.size()),
            rwork.data()
        );

        return info;
    }

    template <class E, class W>
    int heevd(E& A, char jobz, char uplo, W& w)
    {
        XTENSOR_ASSERT(A.dimension() == 2);
        XTENSOR_ASSERT(A.layout() == layout_type::column_major);

        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;

        auto N = A.shape()[0];
        uvector<value_type> work(1);
        uvector<underlying_value_type> rwork(1);
        uvector<blas_index_t> iwork(1);

        int info = cxxlapack::heevd<blas_index_t>(
            jobz,
            uplo,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            w.data(),
            work.data(),
            static_cast<blas_index_t>(-1),
            rwork.data(),
            static_cast<blas_index_t>(-1),
            iwork.data(),
            static_cast<blas_index_t>(-1)
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for heevd.");
        }

        work.resize(std::size_t(std::real(work[0])));
        rwork.resize(std::size_t(rwork[0]));
        iwork.resize(std::size_t(iwork[0]));

        info = cxxlapack::heevd<blas_index_t>(
            jobz,
            uplo,
            static_cast<blas_index_t>(N),
            A.data(),
            stride_back(A),
            w.data(),
            work.data(),
            static_cast<blas_index_t>(work.size()),
            rwork.data(),
            static_cast<blas_index_t>(rwork.size()),
            iwork.data(),
            static_cast<blas_index_t>(iwork.size())
        );

        return info;
    }

    template <class E, class F, class S, std::enable_if_t<!xtl::is_complex<typename E::value_type>::value>* = nullptr>
    int gelsd(E& A, F& b, S& s, blas_index_t& rank, double rcond)
    {
        using value_type = typename E::value_type;

        uvector<value_type> work(1);
        uvector<blas_index_t> iwork(1);

        blas_index_t b_dim = b.dimension() > 1 ? static_cast<blas_index_t>(b.shape().back()) : 1;

        std::size_t m = A.shape()[0];
        std::size_t n = A.shape()[1];

        blas_index_t a_stride = static_cast<blas_index_t>(std::max(std::size_t(1), m));
        blas_index_t b_stride = static_cast<blas_index_t>(std::max(std::max(std::size_t(1), m), n));

        int info = cxxlapack::gelsd<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            b_dim,
            A.data(),
            a_stride,
            b.data(),
            b_stride,
            s.data(),
            rcond,
            rank,
            work.data(),
            static_cast<blas_index_t>(-1),
            iwork.data()
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for gelsd.");
        }

        work.resize(std::size_t(work[0]));
        iwork.resize(std::size_t(iwork[0]));

        info = cxxlapack::gelsd<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            b_dim,
            A.data(),
            a_stride,
            b.data(),
            b_stride,
            s.data(),
            rcond,
            rank,
            work.data(),
            static_cast<blas_index_t>(work.size()),
            iwork.data()
        );

        return info;
    }

    template <class E, class F, class S, std::enable_if_t<xtl::is_complex<typename E::value_type>::value>* = nullptr>
    int gelsd(E& A, F& b, S& s, blas_index_t& rank, double rcond = -1)
    {
        using value_type = typename E::value_type;
        using underlying_value_type = typename value_type::value_type;

        uvector<value_type> work(1);
        uvector<underlying_value_type> rwork(1);
        uvector<blas_index_t> iwork(1);

        blas_index_t b_dim = b.dimension() > 1 ? static_cast<blas_index_t>(b.shape().back()) : 1;
        blas_index_t b_stride = b_dim == 1 ? static_cast<blas_index_t>(b.shape().front()) : stride_back(b);

        int info = cxxlapack::gelsd<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            b_dim,
            A.data(),
            stride_back(A),
            b.data(),
            b_stride,
            s.data(),
            rcond,
            rank,
            work.data(),
            static_cast<blas_index_t>(-1),
            rwork.data(),
            iwork.data()
        );

        if (info != 0)
        {
            XTENSOR_THROW(std::runtime_error, "Could not find workspace size for gelsd.");
        }

        work.resize(std::size_t(std::real(work[0])));
        rwork.resize(std::size_t(rwork[0]));
        iwork.resize(std::size_t(iwork[0]));

        info = cxxlapack::gelsd<blas_index_t>(
            static_cast<blas_index_t>(A.shape()[0]),
            static_cast<blas_index_t>(A.shape()[1]),
            b_dim,
            A.data(),
            stride_back(A),
            b.data(),
            b_stride,
            s.data(),
            rcond,
            rank,
            work.data(),
            static_cast<blas_index_t>(work.size()),
            rwork.data(),
            iwork.data()
        );

        return info;
    }
}

}
#endif
