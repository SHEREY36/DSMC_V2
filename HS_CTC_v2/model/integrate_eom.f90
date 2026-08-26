
	subroutine integrate_eom
	use particleS
	use run_param
	USE CONSTANTS
	
	implicit none
	
	! Rotation
	INTEGER :: I
	DOUBLE PRECISION, DIMENSION(3) :: W
	DOUBLE PRECISION, DIMENSION(3) :: IW, IWxW
	DOUBLE PRECISION :: THT1, THT2, hTHT1, hTHT2
	DOUBLE PRECISION :: q1(3), q10, q2(3), q20
	DOUBLE PRECISION :: q12(3), q120

	! Velocity-Verlet Scheme using half-step for momentum
	! Translation
	DO I = 1,2
		IF(ABS(F(I,1)).LT.SMALL_NUM) F(I,1) = 0.D0
		IF(ABS(F(I,2)).LT.SMALL_NUM) F(I,2) = 0.D0
		IF(ABS(F(I,3)).LT.SMALL_NUM) F(I,3) = 0.D0

		VEL(I,:) = VEL(I,:) + (F(I,:)/MASS)*DT*0.5D0
		POS(I,:) = POS(I,:) + VEL(I,:)*DT
	END DO

	! Rotation
	DO I = 1,2
		IF(ABS(TAU(I,1)).LT.SMALL_NUM) TAU(I,1) = 0.D0
		IF(ABS(TAU(I,2)).LT.SMALL_NUM) TAU(I,2) = 0.D0
		IF(ABS(TAU(I,3)).LT.SMALL_NUM) TAU(I,3) = 0.D0
		
		W(1) = OMEGA(I,1);
		W(2) = OMEGA(I,2);
		W(3) = OMEGA(I,3);

		IW(1) = moI(1)*W(1);
		IW(2) = moI(1)*W(2);
		IW(3) = moI(1)*W(3);

		IWxW(1) = IW(2)*W(3) - IW(3)*W(2);
		IWxW(2) = IW(3)*W(1) - IW(1)*W(3);
		IWxW(3) = IW(1)*W(2) - IW(2)*W(1);
		IWxW(:) = 0.D0

		! Half step rot. velocity
		W(1) = W(1) + OMOI(1)*(TAU(I,1)+IWxW(1))*DT*0.5D0;
		W(2) = W(2) + OMOI(2)*(TAU(I,2)+IWxW(2))*DT*0.5D0;
		W(3) = W(3) + OMOI(3)*(TAU(I,3)+IWxW(3))*DT*0.5D0;

		! Update orientation using half velocity
		! Use quarternions

		OMEGA(I,1) = W(1);
		OMEGA(I,2) = W(2);
		OMEGA(I,3) = W(3);

		!WDOT(1) = DOT_PRODUCT(TAU(I,:),U(I,:))*OMOI(1)
		!WDOT(2) = DOT_PRODUCT(TAU(I,:),UX(I,:))*OMOI(2)
		!WDOT(3) = DOT_PRODUCT(TAU(I,:),UY(I,:))*OMOI(3)
				
		!OMEGA(I,:) = OMEGA(I,:) + WDOT(:)*DT*0.5D0
		
		! Calculate Quarternions

		THT1 = OMEGA(I,2)*dt; hTHT1 = THT1*0.5D0
		THT2 = OMEGA(I,3)*dt; hTHT2 = THT2*0.5D0
		q1 = UX(I,:)*SIN(hTHT1); q10 = COS(hTHT1)
		q2 = UY(I,:)*SIN(hTHT2); q20 = COS(hTHT2)
		q12 = q10*q2 + q20*q1 + CROSSPRDCT(q1,q2)
		q120 = q10*q20 - DOT_PRODUCT(q1,q2)
		
		! Rotate
		U(I,:) = U(I,:)*(q120-DOT_PRODUCT(q12,q12)) + 2.D0*DOT_PRODUCT(q12,U(I,:))*q12&
			+ 2.D0*q120*CROSSPRDCT(q12,U(I,:))
		UX(I,:) = UX(I,:)*(q120-DOT_PRODUCT(q12,q12)) + 2.D0*DOT_PRODUCT(q12,UX(I,:))*q12&
			+ 2.D0*q120*CROSSPRDCT(q12,UX(I,:))
		UY(I,:) = UY(I,:)*(q120-DOT_PRODUCT(q12,q12)) + 2.D0*DOT_PRODUCT(q12,UY(I,:))*q12&
			+ 2.D0*q120*CROSSPRDCT(q12,UY(I,:))
		! Normalize
		U(I,:) = U(I,:)/SQRT(DOT_PRODUCT(U(I,:),U(I,:)))
		UX(I,:) = UX(I,:)/SQRT(DOT_PRODUCT(UX(I,:),UX(I,:)))
		UY(I,:) = UY(I,:)/SQRT(DOT_PRODUCT(UY(I,:),UY(I,:)))
	END DO

	CALL CALC_FORCE_DEM
	
	DO I = 1,2
		IF(ABS(TAU(I,1)).LT.SMALL_NUM) TAU(I,1) = 0.D0;
		IF(ABS(TAU(I,2)).LT.SMALL_NUM) TAU(I,2) = 0.D0;
		IF(ABS(TAU(I,3)).LT.SMALL_NUM) TAU(I,3) = 0.D0;

		IF(ABS(F(I,1)).LT.SMALL_NUM) F(I,1) = 0.D0;
		IF(ABS(F(I,2)).LT.SMALL_NUM) F(I,2) = 0.D0;
		IF(ABS(F(I,3)).LT.SMALL_NUM) F(I,3) = 0.D0;

		VEL(I,:) = VEL(I,:) + (F(I,:)/MASS)*DT*0.5D0;

		W(1) = OMEGA(I,1);
		W(2) = OMEGA(I,2);
		W(3) = OMEGA(I,3);

		IW(1) = moI(1)*W(1);
		IW(2) = moI(2)*W(2);
		IW(3) = moI(3)*W(3);

		IWxW(1) = IW(2)*W(3) - IW(3)*W(2);
		IWxW(2) = IW(3)*W(1) - IW(1)*W(3);
		IWxW(3) = IW(1)*W(2) - IW(2)*W(1);

		W(1) = W(1) + OMOI(1)*(TAU(I,1)+IWxW(1))*DT*0.5D0;
		W(2) = W(2) + OMOI(2)*(TAU(I,2)+IWxW(2))*DT*0.5D0;
		W(3) = W(3) + OMOI(3)*(TAU(I,3)+IWxW(3))*DT*0.5D0;

		OMEGA(I,1) = W(1);
		OMEGA(I,2) = W(2);
		OMEGA(I,3) = W(3);
	END DO

	return
	end subroutine integrate_eom


