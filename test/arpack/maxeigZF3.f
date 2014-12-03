      subroutine maxeigzp(v2,d1,nx,H1)
      implicit none 
      integer           maxn, maxnev, maxncv, ldv
      parameter         (maxn=65536, maxnev=7, maxncv=36, ldv=maxn)

      integer           iparam(11), ipntr(14)
      logical           select(maxncv)
      Complex*16
     &                  ax(maxn), d(maxncv),d1(maxnev),
     &                  v(ldv,maxncv), workd(3*maxn),H1(625),
     &                  workev(3*maxncv), resid(maxn),
     &                  workl(3*maxncv*maxncv+5*maxncv),v2(4*ldv)
      Double precision
     &                  rwork(maxncv), rd(maxncv,3)

      character         bmat*1, which*2
      integer           ido, n, nx, nev, ncv, lworkl, info,i,
     &                  ierr, nconv, maxitr, ishfts, mode, j,k
      Complex*16
     &                  sigma, tym
      Double precision
     &                  tol
      logical           rvec

      Double precision
     &                  dznrm2, dlapy2
      external          dznrm2, zaxpy, dlapy2, avcp

      n     = nx
      nev   = 7
      ncv   = 36

      bmat  = 'I'
      which = 'LM'

      lworkl  = 3*ncv**2+5*ncv
      tol    = 0.0
      ido    = 0
      info   = 1

      ishfts = 1
      maxitr = 90000
      mode   = 1

      iparam(1) = ishfts
      iparam(3) = maxitr
      iparam(7) = mode
       
      do j=1, n
       resid(j)=(1.0,0.0)
      enddo

 10   continue

         call znaupd ( ido, bmat, n, which, nev, tol, resid, ncv,
     &        v, ldv, iparam, ipntr, workd, workl, lworkl,
     &        rwork,info )

         if (ido .eq. -1 .or. ido .eq. 1) then

            call avcp (workd(ipntr(1)), workd(ipntr(2)),H1)

            go to 10

         end if

      if ( info .lt. 0 ) then

         print *, ' '
         print *, ' Error with _naupd, info = ', info
         print *, ' Check the documentation of _naupd'
         print *, ' '

      else

         rvec = .true.

         call zneupd (rvec, 'A', select, d, v, ldv, sigma,
     &        workev, bmat, n, which, nev, tol, resid, ncv,
     &        v, ldv, iparam, ipntr, workd, workl, lworkl,
     &        rwork, ierr)

         if ( ierr .ne. 0) then

             print *, ' '
             print *, ' Error with _neupd, info = ', ierr
             print *, ' Check the documentation of _neupd. '
             print *, ' '
c
         else
c
             nconv = iparam(5)
             do 20 j=1, nconv

                call avcp ( v(1,j), ax,H1)
                call zaxpy(n, -d(j), v(1,j), 1, ax, 1)
                rd(j,1) = dble(d(j))
                rd(j,2) = dimag(d(j))
                rd(j,3) = dznrm2(n, ax, 1)
                rd(j,3) = rd(j,3) / dlapy2(rd(j,1),rd(j,2))
 20          continue

          end if

      end if

c      do 30 i=1, nev
c          k=(i-1)*n ->v2(k+j)
      do 40 j=1, n
      	    v2(j)=v(j,nev)
 40   continue
c 30   continue
 
c      do i=2, nev
      d1=d(nev)
c      enddo

      return
      end
c
c==========================================================================
c
      subroutine ortog(vP,nx,mx)
      implicit none 
      integer           maxn, maxldv
      parameter       (maxn=655, maxldv=10*maxn)
      
      integer         n, nx, lda, lwork, info,i,j,m,k,i1,mx
      Complex*16      A(maxn,maxn), vP(maxn*maxn), Tau(maxn), 
     &                work(maxldv)
      external        zgeqrf, zungqr
      intrinsic       int
     
      n=nx
      lda=mx
      m=mx
      k=nx
      do 10 i=1, m
         do 20 j=1, n
            i1=(i-1)*n+j
      	    A(i,j)=vP(i1)
 20   continue
 10   continue
     
      lwork = -1
      call zgeqrf(m, n, A, lda, Tau, work, lwork, info)
      lwork = int(work(1))
      call zgeqrf(m, n, A, lda, Tau, work, lwork, info)
  
      lwork = -1;
      call zungqr(m, n, k, A, lda, Tau, work, lwork, info);
      lwork = int(work(1))
      call zungqr(m, n, k, A, lda, Tau, work, lwork, info);
 

      do 30 i=1, m
         do 40 j=1, n
            i1=(i-1)*n+j
      	    vP(i1) = A(j,i)
 40   continue
 30   continue
     
      return
      end
      
c==========================================================================
c
      subroutine maxeigzrop(v2,ww,nx,ne,nc)
      implicit none 
      integer           maxn, maxnev, maxncv, ldv
      parameter         (maxn=125, maxnev=35, maxncv=125, ldv=maxn)
c                             81                    81
      integer           iparam(11), ipntr(14)
      logical           select(maxncv)
      Complex*16
     &                  ax(maxn), d(maxncv),v2(maxn,maxncv),
     &                  v(ldv,maxncv), workd(3*maxn),ww(maxnev),
     &                  workev(3*maxncv), resid(maxn),
     &                  workl(3*maxncv*maxncv+5*maxncv)
      Double precision
     &                  rwork(maxncv), rd(maxncv,3)

      character         bmat*1, which*2
      integer           ido, n, nx, nev, ncv, lworkl, info,mnoz,ne,
     &                  ierr, nconv, maxitr, ishfts, mode, j,ind,i,nc
      Complex*16
     &                  sigma, d1
      Double precision
     &                  tol
      logical           rvec

      Double precision
     &                  dznrm2, dlapy2
      external         dznrm2, zaxpy, dlapy2, avcrop

      n     = nx
      nev   = ne
      ncv   = nc

      bmat  = 'I'
      which = 'LM'

      lworkl  = 3*ncv**2+5*ncv
      tol    = 0.0
      ido    = 0
      info   = 1

      ishfts = 1
      maxitr = 90000
      mode   = 1

      iparam(1) = ishfts
      iparam(3) = maxitr
      iparam(7) = mode

      do j=1, n
         resid(j)=(1.0,0.0)
      enddo

 10   continue

         call znaupd ( ido, bmat, n, which, nev, tol, resid, ncv,
     &        v, ldv, iparam, ipntr, workd, workl, lworkl,
     &        rwork,info )

         if (ido .eq. -1 .or. ido .eq. 1) then

            call avcrop (workd(ipntr(1)), workd(ipntr(2)),n)

            go to 10

         end if

      if ( info .lt. 0 ) then

         print *, ' '
         print *, ' Error with _naupd, info = ', info
         print *, ' Check the documentation of _naupd'
         print *, ' '

      else

         rvec = .true.

         call zneupd (rvec, 'A', select, d, v, ldv, sigma,
     &        workev, bmat, n, which, nev, tol, resid, ncv,
     &        v, ldv, iparam, ipntr, workd, workl, lworkl,
     &        rwork, ierr)

         if ( ierr .ne. 0) then

             print *, ' '
             print *, ' Error with _neupd, info = ', ierr
             print *, ' Check the documentation of _neupd. '
             print *, ' '
c
         else
c
             nconv = iparam(5)
             do 20 j=1, nconv

                call avcrop ( v(1,j), ax,n)
                call zaxpy(n, -d(j), v(1,j), 1, ax, 1)
                rd(j,1) = dble(d(j))
                rd(j,2) = dimag(d(j))
                rd(j,3) = dznrm2(n, ax, 1)
                rd(j,3) = rd(j,3) / dlapy2(rd(j,1),rd(j,2))
 20          continue

          end if

      end if

      do 30 i=1, nev
        do 40 j=1, n
      	    v2(j,i)=v(j,i)
 40   continue
 30   continue
 
      do i=1, nev
        ww(i) = d(i)
      enddo    

      return
      end
c
c==========================================================================
c
