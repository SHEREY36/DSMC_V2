
	subroutine calc_force_dem
	USE PARTICLES
	USE CONSTANTS
	USE OUTPUT
	implicit none
	DOUBLE PRECISION, DIMENSION(3) :: RHO1, RHO2, E21
	DOUBLE PRECISION, DIMENSION(3) :: VR, OMEGAI, OMEGAJ, V_ROT, VREL_CONTACT
	DOUBLE PRECISION :: VRN
	DOUBLE PRECISION :: DISTSQ, DN
	DOUBLE PRECISION, DIMENSION(3) :: FN, F_TMP, TAU_FORCE
	! temporary storage of torque
        DOUBLE PRECISION :: TAU_TMP(3,2)
	DOUBLE PRECISION :: LAMBDA_TMP, MU_TMP
	INTEGER :: I, J

	F(:,:) = 0.D0; TAU(:,:) = 0.D0;

	I=1; J=2;

	CALL OVERLAP_PP(DISTSQ, RHO1, RHO2, E21, LAMBDA_TMP, MU_TMP)
	F_TMP(:) = 0.0D0

	! Translational relative velocity (particle j relative to particle i)
	VR = VEL(2,:) - VEL(1,:)

	! Contact-point rotational velocity.
	! For smooth spherocylinders the twist about the symmetry axis does not
	! contribute, so only the components perpendicular to U are retained here.
	OMEGAI = OMEGA(I,2)*UX(I,:) + OMEGA(I,3)*UY(I,:)
	OMEGAJ = OMEGA(J,2)*UX(J,:) + OMEGA(J,3)*UY(J,:)
	V_ROT = CROSSPRDCT(OMEGAJ, RHO2) - CROSSPRDCT(OMEGAI, RHO1)
	VREL_CONTACT = VR !+ V_ROT

	! Normal approach speed at the contact point.
	! Positive VRN means the surfaces approach along the contact normal E21.
	VRN = -(DOT_PRODUCT(VREL_CONTACT, E21))

	CONTACT = .FALSE.
	IF(DISTSQ.GT.(DIA-SMALL_NUM)**2.D0) RETURN

	CONTACT = .TRUE.
	DN = DIA - SQRT(DISTSQ)

	! Calculate the normal contact force
	! Force vector points from i to j
	! Damping is skipped during the elastic replay pass (CN=0 equivalent)
	IF (ELASTIC_PASS) THEN
		FN(:) = KN*DN*E21
	ELSE
		FN(:) = (KN*DN*E21) + (CN*VRN*E21)
	END IF

	F_TMP(:) = FN(:)

	F(I,:) = F(I,:) - F_TMP(:)
        F(J,:) = F(J,:) + F_TMP(:)

	! Torques felt by each particle
	! Use temporary force, not FC; otherwise, overcounting torques
	! In cases where mulitple contacts exist
        TAU_force(:) = CROSSPRDCT(RHO1, -F_TMP(:))
        TAU_TMP(1,1) = DOT_PRODUCT(TAU_FORCE,U(I,:))
        TAU_TMP(2,1) = DOT_PRODUCT(TAU_FORCE,UX(I,:))
        TAU_TMP(3,1) = DOT_PRODUCT(TAU_FORCE,UY(I,:))

        TAU_force(:) = CROSSPRDCT(RHO2, F_TMP(:))
        TAU_TMP(1,2) = DOT_PRODUCT(TAU_FORCE,U(J,:))
        TAU_TMP(2,2) = DOT_PRODUCT(TAU_FORCE,UX(J,:))
        TAU_TMP(3,2) = DOT_PRODUCT(TAU_FORCE,UY(J,:))

        TAU(1,:)  = TAU(1,:) + TAU_TMP(:,1)
        TAU(2,:)  = TAU(2,:) + TAU_TMP(:,2)

	IF(.NOT.HIT) THEN
		HIT = .TRUE.
		CALL PROJECTED_AREA()
		! Record normalised contact-point positions (already computed above)
		contact_lambda = LAMBDA_TMP
		contact_mu     = MU_TMP
		! Orientation descriptors at first contact
		CALL CALC_ORIENTATION_AT_CONTACT(E21, VRN)
	END IF
	return
	end subroutine calc_force_dem
!-------------------------------------------------------------------
	SUBROUTINE OVERLAP_PP(DSQ, RHO1, RHO2, E21, LAMBDA_OUT, MU_OUT)
	USE CONSTANTS, ONLY: SMALL_NUM
	USE PARTICLES
	IMPLICIT NONE
	DOUBLE PRECISION :: R12(3), R12SQ
	DOUBLE PRECISION :: C21(3)
	DOUBLE PRECISION :: U1DOT, U2DOT, UDOT
	DOUBLE PRECISION :: CC, oCC
	DOUBLE PRECISION :: LAMBDA, MU
	DOUBLE PRECISION :: V1, V2
	DOUBLE PRECISION, INTENT(OUT) :: DSQ, RHO1(3), RHO2(3), E21(3)
	DOUBLE PRECISION, INTENT(OUT) :: LAMBDA_OUT, MU_OUT
	DOUBLE PRECISION :: C1(3), C2(3)

	R12 = POS(2,:)-POS(1,:); R12SQ = DOT_PRODUCT(R12,R12)
	U1DOT = DOT_PRODUCT(U(1,:),R12); U2DOT = DOT_PRODUCT(U(2,:),R12)
	UDOT = DOT_PRODUCT(U(1,:),U(2,:)); CC = 1.D0 - UDOT**2.D0
	IF(CC.LT.SMALL_NUM) THEN
		DSQ = R12SQ - U1DOT**2.D0 + (MAX(0.D0,ABS(U1DOT)-hLCYL))**2.D0
		IF(ABS(U1DOT).GT.SMALL_NUM) THEN
			LAMBDA = SIGN(hLCYL,U1DOT)
			MU = LAMBDA*UDOT-U1DOT
			IF(ABS(MU)>hLCYL) MU = SIGN(hLCYL,MU)
		ELSE
			LAMBDA = 0.D0; MU = 0.D0
		END IF
	ELSE
		oCC = 1.D0/CC
		LAMBDA = (U1DOT-UDOT*U2DOT)*oCC
		MU = (-U2DOT+UDOT*U1DOT)*oCC
		V1 = ABS(LAMBDA)-hLCYL; V2 = ABS(MU)-hLCYL
		IF(V1.GT.0.D0.OR.V2.GT.0.D0) THEN
			IF(V1.GT.V2) THEN
				LAMBDA = SIGN(hLCYL,LAMBDA)
				MU = LAMBDA*UDOT-U2DOT
				IF(ABS(MU).GT.hLCYL) MU = SIGN(hLCYL,MU)
			ELSE
				MU = SIGN(hLCYL,MU)
				LAMBDA = MU*UDOT+U1DOT
				IF(ABS(LAMBDA).GT.hLCYL) LAMBDA = SIGN(hLCYL,LAMBDA)
			END IF
		END IF
		DSQ = R12SQ + LAMBDA**2.D0 + MU**2.D0 &
			- 2.D0*LAMBDA*MU*UDOT + 2.D0*MU*U2DOT &
			- 2.D0*LAMBDA*U1DOT
	END IF
	RHO1 = LAMBDA*U(1,:)
	RHO2 = MU*U(2,:);
	C1 = POS(1,:) + RHO1;
	C2 = POS(2,:) + RHO2
	C21 = C2-C1;
	E21 = C21/SQRT(DSQ)

	! Pass back normalised contact-point positions
	IF (hLCYL > SMALL_NUM) THEN
		LAMBDA_OUT = LAMBDA / hLCYL
		MU_OUT     = MU     / hLCYL
	ELSE
		! A sphere (AR=1) has no cylindrical axis coordinate.
		LAMBDA_OUT = 0.D0
		MU_OUT     = 0.D0
	END IF

	return
	END SUBROUTINE OVERLAP_PP
!-------------------------------------------------------------------
	SUBROUTINE PROJECTED_AREA()
	USE PARTICLES
	USE OUTPUT
	IMPLICIT NONE
	DOUBLE PRECISION, DIMENSION(3) :: V12, R12, RPOSF,RVN12,RN12
	DOUBLE PRECISION, DIMENSION(3) :: U1, U2

	V12 = VEL(2,:) - VEL(1,:); V12 = V12/SQRT(DOT_PRODUCT(V12,V12))
	R12 = POS(2,:) - POS(1,:); R12 = R12/SQRT(DOT_PRODUCT(R12,R12))
	U1 = U(1,:) - DOT_PRODUCT(U(1,:),V12)*V12
	U2 = U(2,:) - DOT_PRODUCT(U(2,:),V12)*V12
	PROJ_AREA = &
		ABS(DOT_PRODUCT(U1,R12)) + ABS(DOT_PRODUCT(U2,R12))
	RVN12 = VEL(2,:) - VEL(1,:);
	RN12 = POS(2,:) - POS(1,:)
	RPOSF = RN12 - DOT_PRODUCT(RVN12,RN12)&
		*RVN12/DOT_PRODUCT(RVN12,RVN12)
	b_contact = SQRT(DOT_PRODUCT(RPOSF,RPOSF))/(LCYL+DIA)
	return
	END SUBROUTINE PROJECTED_AREA
!-------------------------------------------------------------------
	SUBROUTINE CALC_ORIENTATION_AT_CONTACT(E21_in, VRN_in)
	USE PARTICLES, ONLY: U, UX, UY, VEL, OMEGA, MASS, POS
	USE OUTPUT
	USE CONSTANTS, ONLY: SMALL_NUM
	IMPLICIT NONE
	DOUBLE PRECISION, INTENT(IN) :: E21_in(3), VRN_in
	DOUBLE PRECISION :: vhat(3), vnorm
	DOUBLE PRECISION :: r12_fc(3), r12_fc_hat(3), r12_fc_norm
	DOUBLE PRECISION :: a12, a1n, a2n, a1v, a2v

	! Unit approach-velocity vector (pre-collision, from output module)
	vnorm = SQRT(DOT_PRODUCT(VREL0, VREL0))
	IF (vnorm > 1.0D-30) THEN
		vhat = VREL0 / vnorm
	ELSE
		vhat = E21_in   ! degenerate: use contact normal as fallback
	END IF

	r12_fc = POS(2,:) - POS(1,:)
	r12_fc_norm = SQRT(DOT_PRODUCT(r12_fc, r12_fc))
	IF (r12_fc_norm > SMALL_NUM .AND. vnorm > SMALL_NUM) THEN
		r12_fc_hat  = r12_fc / r12_fc_norm
		mu_in       = ABS(DOT_PRODUCT(r12_fc_hat, vhat))
		eij_contact = r12_fc_hat
	ELSE
		mu_in       = 0.0D0
		eij_contact = 0.0D0
	END IF

	! Dot products (signed; headless symmetry handled in S2 via squaring)
	a12 = DOT_PRODUCT(U(1,:), U(2,:))
	a1n = DOT_PRODUCT(U(1,:), E21_in)
	a2n = DOT_PRODUCT(U(2,:), E21_in)
	a1v = DOT_PRODUCT(U(1,:), vhat)
	a2v = DOT_PRODUCT(U(2,:), vhat)

	! Nematic order parameters: P2(cos theta) = (3cos^2(theta) - 1)/2
	! Range [-0.5, 1.0]; S2=1 perfectly aligned, S2=-0.5 perfectly perpendicular
	S2_pair = 0.5D0*(3.0D0*a12**2 - 1.0D0)
	S2_1n   = 0.5D0*(3.0D0*a1n**2 - 1.0D0)
	S2_2n   = 0.5D0*(3.0D0*a2n**2 - 1.0D0)
	S2_1v   = 0.5D0*(3.0D0*a1v**2 - 1.0D0)
	S2_2v   = 0.5D0*(3.0D0*a2v**2 - 1.0D0)

	! Raw unsigned cosines
	cos_u1_n = ABS(a1n);  cos_u2_n = ABS(a2n)
	cos_u1_v = ABS(a1v);  cos_u2_v = ABS(a2v)

	! Signed u1.u2: captures chirality of the pair configuration
	u1u2_dot = a12

	! Pre-collision orientation vectors (full 3D)
	U1_pre = U(1,:);  U2_pre = U(2,:)
	C1_pre = VEL(1,:); C2_pre = VEL(2,:)
	OMEGA1_lab_pre = OMEGA(1,1)*U(1,:) + OMEGA(1,2)*UX(1,:) + OMEGA(1,3)*UY(1,:)
	OMEGA2_lab_pre = OMEGA(2,1)*U(2,:) + OMEGA(2,2)*UX(2,:) + OMEGA(2,3)*UY(2,:)
	contact_normal = E21_in

	! Normal-direction translational energy at contact
	E_n_pre = 0.5D0 * MASS * VRN_in**2

	return
	END SUBROUTINE CALC_ORIENTATION_AT_CONTACT
