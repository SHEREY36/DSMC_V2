
	SUBROUTINE INIT_PART

! Global variables
!------------------------------------------------------------------//
	use particleS
	USE CONSTANTS
	use run_param
	use output
	use rng_mod
!$ use omp_lib

	implicit none
	INTEGER :: I
	DOUBLE PRECISION :: RR
	DOUBLE PRECISION :: PHI, THETA
	DOUBLE PRECISION :: VS
	DOUBLE PRECISION :: ROT(3,3), G_VEC(3), V_CM(3), BASIS_NORM

	double precision, dimension(3) :: v1com, v2com, vcom
	double precision, dimension(3) :: RVEL, RPOS

	! Sample Position
	POS(1,:) = (/0.D0, 0.D0, 0.D0/)
	POS(2,1) = BMAX
	DO I = 2,3
		RR = RNG_UNIFORM()
		POS(2,I) = BMAX*RR
	END DO
	
	! Sample Orientation Vectors
	DO I = 1,2
		RR = RNG_UNIFORM(); PHI = ACOS(RR*2.D0 - 1.D0)
		RR = RNG_UNIFORM(); THETA = 2.D0*RR*PI
		U(I,:) = (/SIN(PHI)*COS(THETA), SIN(PHI)*SIN(THETA), COS(PHI)/)
		U(I,:) = U(I,:)/SQRT(DOT_PRODUCT(U(I,:),U(I,:)))
		DO
			RR = RNG_UNIFORM(); PHI = ACOS(RR*2.D0 - 1.D0)
			RR = RNG_UNIFORM(); THETA = 2.D0*RR*PI
			UX(I,:) = (/SIN(PHI)*COS(THETA), SIN(PHI)*SIN(THETA), COS(PHI)/)
			UX(I,:) = UX(I,:) - DOT_PRODUCT(UX(I,:),U(I,:))*U(I,:)
			BASIS_NORM = SQRT(DOT_PRODUCT(UX(I,:),UX(I,:)))
			IF (BASIS_NORM > SMALL_NUM) EXIT
		END DO
		UX(I,:) = UX(I,:)/BASIS_NORM
		UY(I,:) = CROSSPRDCT(UX(I,:),U(I,:))
		UY(I,:) = UY(I,:)/SQRT(DOT_PRODUCT(UY(I,:),UY(I,:)))
	END DO

! Constant Temperature
	VEL(:,:) = 0.D0
	OMEGA(:,:) = 0.D0
	IF(ToE.EQ.'T') THEN
		! The collision-weighted relative-speed law is
		! p(g) proportional to g^3 exp(-g^2/4). Therefore g^2/4 is
		! Gamma(shape=2, scale=1), sampled exactly as two exponentials.
		VS = 2.D0*SQRT(-LOG(MAX(RNG_UNIFORM()*RNG_UNIFORM(), TINY(1.D0))))
		G_VEC = (/VS*SQRT(kTm), 0.D0, 0.D0/)

		! For two equal Maxwellian particles, Var(V_cm,k)=T/(2m).
		DO I = 1,3
			V_CM(I) = SQRT(0.5D0*kTm)*RNG_NORMAL()
		END DO
		VEL(1,:) = V_CM + 0.5D0*G_VEC
		VEL(2,:) = V_CM - 0.5D0*G_VEC

		! The two transverse body-frame spin components are independent,
		! untruncated Maxwellian normal variates.
		DO I = 1,2
			OMEGA(I,2) = SQRT(kTI)*RNG_NORMAL()
			OMEGA(I,3) = SQRT(kTI)*RNG_NORMAL()
		END DO
	ELSE IF(ToE.EQ.'E') THEN
		RR = RNG_UNIFORM()
		OMEGA(1,2) = SQEr*RR; OMEGA(1,3) = SQEr*(1.D0-RR)
		RR = RNG_UNIFORM()
		OMEGA(2,2) = SQEr*RR; OMEGA(2,3) = SQEr*(1.D0-RR)
		VEL(1,1) = SQEk
	ELSE
		write(*,*) 'Particle init not defined'
		stop
	END IF

	! Rotate the complete canonical collision by a Haar-uniform SO(3)
	! rotation. This preserves the trajectory while restoring laboratory-frame
	! isotropy required by the tensor/vector score projections.
	CALL RANDOM_ROTATION_MATRIX(ROT)
	DO I = 1,2
		POS(I,:) = MATMUL(ROT, POS(I,:))
		VEL(I,:) = MATMUL(ROT, VEL(I,:))
		U(I,:)   = MATMUL(ROT, U(I,:))
		UX(I,:)  = MATMUL(ROT, UX(I,:))
		UY(I,:)  = MATMUL(ROT, UY(I,:))
	END DO
	
	! Initialize forces
	F(:,:) = 0.D0
	TAU(:,:) = 0.D0
	
		! Record initial conditions
	RPOS = POS(2,:) - POS(1,:); RVEL = VEL(2,:) - VEL(1,:)
	RPOS = RPOS - DOT_PRODUCT(RVEL,RPOS)&
		*RVEL/DOT_PRODUCT(RVEL,RVEL)
	b_impact = SQRT(DOT_PRODUCT(RPOS,RPOS))
	TRY_C1 = VEL(1,:); TRY_C2 = VEL(2,:)
	TRY_U1 = U(1,:); TRY_U2 = U(2,:)
	TRY_W1 = OMEGA(1,1)*U(1,:) + OMEGA(1,2)*UX(1,:) + OMEGA(1,3)*UY(1,:)
	TRY_W2 = OMEGA(2,1)*U(2,:) + OMEGA(2,2)*UX(2,:) + OMEGA(2,3)*UY(2,:)
	TRY_IMPACT = RPOS/MAX(LCYL + DIA, SMALL_NUM)
	!TMEAN = TMEAN + MASS*(VEL(1,1)*VEL(1,1))
	!RMEAN = RMEAN + MOI(2)*(OMEGA(1,2)*OMEGA(1,2)+&
	!	OMEGA(1,3)*OMEGA(1,3)+&
	!	OMEGA(2,2)*OMEGA(2,2)+&
	!	OMEGA(2,3)*OMEGA(2,3))
	!write(100,'(2(D10.4,2X))') MASS*(VEL(1,1)*VEL(1,1)),&
	!	MOI(2)*(OMEGA(1,2)*OMEGA(1,2)+OMEGA(1,3)*OMEGA(1,3)+&
	!	OMEGA(2,2)*OMEGA(2,2)+OMEGA(2,3)*OMEGA(2,3))
	!flush(100)
	!write(*,*) TMEAN/(3.D0*DBLE(NTRY+1)), RMEAN/(2.D0*DBLE(NTRY+1))

	VCOM = (VEL(1,:) + VEL(2,:))*0.5D0
	V1COM = VEL(1,:) - VCOM; V2COM = VEL(2,:) - VCOM
	E0 = MASS*(DOT_PRODUCT(V1COM,V1COM) + DOT_PRODUCT(V2COM,V2COM))
	E0 = E0 + &
		moI(2)*(DOT_PRODUCT(OMEGA(1,:),OMEGA(1,:)) +&
		DOT_PRODUCT(OMEGA(2,:),OMEGA(2,:)))
        Er_00 = moI(2)*(DOT_PRODUCT(OMEGA(1,:),OMEGA(1,:)) +&
		DOT_PRODUCT(OMEGA(2,:),OMEGA(2,:)))
	Er_1 = moI(2)*DOT_PRODUCT(OMEGA(1,:),OMEGA(1,:))
	Er_2 = moI(2)*DOT_PRODUCT(OMEGA(2,:),OMEGA(2,:))
        Et_00 = MASS*(DOT_PRODUCT(V1COM,V1COM) + DOT_PRODUCT(V2COM,V2COM))
	
	
	dt = TCOLL/DT_DIVISOR
	! For calls to outputs
	VREL0 = VEL(2,:) - VEL(1,:)
	WREL0 = OMEGA(2,:) - OMEGA(1,:)
	mu_in = 0.D0
	HIT = .FALSE.
	CONTACT = .FALSE.; NPHIT = 0
	RETURN
	end subroutine INIT_PART
