      subroutine maxeigp2(resid,v2,d1,d2,nn)
      implicit none
      integer           maxn, maxnev, maxncv, ldv
      parameter         (maxn=65536, maxnev=4, maxncv=16, ldv=maxn)

      integer           iparam(11), ipntr(14)
      logical           select(maxncv)
      Double precision
     &                  ax(maxn), d(maxncv,3), resid(maxn),
     &                  v(ldv,maxncv), workd(3*maxn),
     &                  workev(3*maxncv),
     &                  workl(3*maxncv*maxncv+6*maxncv),v2(maxn)

      character         bmat*1, which*2
      integer           ido, n, nn, nev, ncv, lworkl, info, j,
     &                  ierr, nconv, maxitr, ishfts, mode
      Double precision
     &                  tol, sigmar, sigmai, d1, d2
      logical           first, rvec

      Double precision
     &                  zero
      parameter         (zero = 0.0D+0)

      Double precision
     &                  dlapy2, dnrm2
      external          dlapy2, dnrm2, daxpy, avc

      intrinsic         abs

      n     = nn
      nev   = 1
      ncv   = 16 
 
      bmat  = 'I'
      which = 'LM'

      lworkl  = 3*ncv**2+6*ncv 
      tol    = zero 
      ido    = 0
      info   = 0

      ishfts = 1
      maxitr = 9000
      mode   = 1

      iparam(1) = ishfts
      iparam(3) = maxitr 
      iparam(7) = mode

 10   continue

         call dnaupd ( ido, bmat, n, which, 8, tol, resid, 
     &        ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, 
     &        info )

         if (ido .eq. -1 .or. ido .eq. 1) then

            call avc (n, workd(ipntr(1)), workd(ipntr(2)))

            go to 10

      end if 

      if ( info .lt. 0 ) then

         print *, ' '
         print *, ' Error with _naupd, info = ', info
         print *, ' Check the documentation of _naupd'
         print *, ' '

      else 

         rvec = .true.

         call dneupd ( rvec, 'A', select, d, d(1,2), v, ldv, 
     &        sigmar, sigmai, workev, bmat, n, which, nev, tol, 
     &        resid, ncv, v, ldv, iparam, ipntr, workd, workl,
     &        lworkl, ierr )

         if ( ierr .ne. 0) then

             print *, ' '
             print *, ' Error with _neupd, info = ', ierr
             print *, ' Check the documentation of _neupd. '
             print *, ' '

         else 

             first  = .true.
             nconv  = iparam(5)
             do 20 j=1, nconv

                if (d(j,2) .eq. zero)  then

                   call avc(n, v(1,j), ax)
                   call daxpy(n, -d(j,1), v(1,j), 1, ax, 1)
                   d(j,3) = dnrm2(n, ax, 1)
                   d(j,3) = d(j,3) / abs(d(j,1))

                else if (first) then

                   call avc(n, v(1,j), ax)
                   call daxpy(n, -d(j,1), v(1,j), 1, ax, 1)
                   call daxpy(n, d(j,2), v(1,j+1), 1, ax, 1)
                   d(j,3) = dnrm2(n, ax, 1)
                   call avc(n, v(1,j+1), ax)
                   call daxpy(n, -d(j,2), v(1,j), 1, ax, 1)
                   call daxpy(n, -d(j,1), v(1,j+1), 1, ax, 1)
                   d(j,3) = dlapy2( d(j,3), dnrm2(n, ax, 1) )
                   d(j,3) = d(j,3) / dlapy2(d(j,1),d(j,2))
                   d(j+1,3) = d(j,3)
                   first = .false.
                else
                   first = .true.
                end if

 20          continue

          end if
c
      end if
c
      d1=d(1,1)
      d2=d(1,2)

      return
      end
c 
c==========================================================================
c
c      subroutine av (n, v, w, A)
c      implicit none
c      integer           n, j, i
c      Double precision         
c     &                  v(n), w(n), A(n,n)
c
c        w<--OP*v
c
c      do 10 i=1, n
c        w(i)=0
c        do 20 j=1, n
c	      w( i ) = w( i ) + v( j )*A( j,i )
c
c	      A( j + (i-1)*n )
c
c  20  continue
c  10  continue

c
c      return
c      end
c=========================================================================

