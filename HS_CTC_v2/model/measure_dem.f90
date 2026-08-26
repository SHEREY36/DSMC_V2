
	subroutine measure_dem(event_id, try_index)

	use run_param
	use particles
	use output
	use, intrinsic :: iso_fortran_env, only: real64

	implicit none
	integer, intent(in) :: event_id
	integer, intent(in) :: try_index
	double precision :: vrelf(3), wrelf(3)
	double precision, dimension(3) :: v1com, v2com, vcom
	double precision :: Ef(2)
	double precision :: chi, psi, Er10, Er20, Er1_prime, Er2_prime
	double precision :: b_out, vrelf_unit(3), rpos12(3), rposf_out(3), vrelf_norm
	double precision :: vrel0_unit(3), vrel0_norm
	double precision :: delta_E_tr, delta_E_rot, delta_E_total, f_tr
	double precision :: elastic_rel_error, angle_cos
	double precision :: omega1_post(3), omega2_post(3)
	real(real64) :: outcome_values(N_OUTCOME_REAL)

	! Scattering Angle
	VRELF = VEL(2,:) - VEL(1,:)
	chi = SQRT(DOT_PRODUCT(VRELF,VRELF))
	chi = SQRT(DOT_PRODUCT(VREL0,VREL0))*chi
	IF (chi > 1.D-30) THEN
		angle_cos = MAX(-1.D0, MIN(1.D0, DOT_PRODUCT(VRELF,VREL0)/chi))
		chi = ACOS(angle_cos)
	ELSE
		chi = 0.D0
	END IF

	! Scattering Angle for angular velocity W
	WRELF = OMEGA(2,:) - OMEGA(1,:)
	psi = SQRT(DOT_PRODUCT(WRELF,WRELF))
	psi = SQRT(DOT_PRODUCT(WREL0,WREL0))*psi
	IF (psi > 1.D-30) THEN
		angle_cos = MAX(-1.D0, MIN(1.D0, DOT_PRODUCT(WRELF,WREL0)/psi))
		psi = ACOS(angle_cos)
	ELSE
		psi = 0.D0
	END IF

	! Post-Collision Energy
	VCOM = (VEL(2,:)+VEL(1,:))*0.5D0
	V1COM = VEL(1,:) - VCOM; V2COM = VEL(2,:) - VCOM
	Ef(1) = MASS*(DOT_PRODUCT(V1COM,V1COM) + DOT_PRODUCT(V2COM,V2COM))
	Ef(2) = MOI(2)*(DOT_PRODUCT(OMEGA(1,2:3),OMEGA(1,2:3)) + &
		DOT_PRODUCT(OMEGA(2,2:3),OMEGA(2,2:3)))

	! --- Elastic pass: store reference energy and return early ---
	IF (ELASTIC_PASS) THEN
		Et_f_elastic = Ef(1)
		Er_f_elastic = Ef(2)
		RETURN
	END IF

	! --- Inelastic pass below ---

	Er10 = Er_1
	Er20 = Er_2
	Er1_prime = MOI(2)*DOT_PRODUCT(OMEGA(1,2:3),OMEGA(1,2:3))
	Er2_prime = MOI(2)*DOT_PRODUCT(OMEGA(2,2:3),OMEGA(2,2:3))

	! Post-collision orientation vectors
	U1_post = U(1,:)
	U2_post = U(2,:)

	! Positive routing convention: energy present after the elastic replay but
	! removed by the inelastic replay. Using the elastic final state also
	! subtracts the integrator's conservative baseline error.
	delta_E_tr    = Et_f_elastic - Ef(1)
	delta_E_rot   = Er_f_elastic - Ef(2)
	delta_E_total = delta_E_tr + delta_E_rot
	elastic_rel_error = (Et_f_elastic + Er_f_elastic - E0)/MAX(E0, 1.D-30)
	IF (delta_E_total > 1.0D-30*MAX(E0, 1.D0)) THEN
		f_tr = delta_E_tr / delta_E_total
	ELSE
		f_tr = -999.0D0   ! sentinel: elastic collision, f_tr undefined
	END IF

	! b_out: outgoing impact parameter (corrected b_contact)
	! Computed from post-collision outgoing velocities/positions
	vrelf_norm = SQRT(DOT_PRODUCT(VRELF, VRELF))
	IF (vrelf_norm > 1.0D-30) THEN
		vrelf_unit = VRELF / vrelf_norm
		rpos12     = POS(2,:) - POS(1,:)
		rposf_out  = rpos12 - DOT_PRODUCT(rpos12, vrelf_unit) * vrelf_unit
		b_out      = SQRT(DOT_PRODUCT(rposf_out, rposf_out)) / (LCYL + DIA)
	ELSE
		vrelf_unit = 0.0D0
		b_out      = 0.0D0
	END IF

	! Pre-collision relative velocity unit vector
	vrel0_norm = SQRT(DOT_PRODUCT(VREL0, VREL0))
	IF (vrel0_norm > 1.0D-30) THEN
		vrel0_unit = VREL0 / vrel0_norm
	ELSE
		vrel0_unit = 0.0D0
	END IF

	! Buffer legacy outputs when requested.
	IF (WRITE_LEGACY) THEN
	buffer_idx        = buffer_idx        + 1
	buffer_ftr_idx    = buffer_ftr_idx    + 1
	buffer_orient_idx = buffer_orient_idx + 1
	buffer_uvec_idx   = buffer_uvec_idx   + 1

	chi_buffer(buffer_idx, 1)  = b_impact
	chi_buffer(buffer_idx, 2)  = chi
	chi_buffer(buffer_idx, 3)  = psi
	chi_buffer(buffer_idx, 4)  = mu_in
	chi_buffer(buffer_idx, 5)  = eij_contact(1)   ! eij_x
	chi_buffer(buffer_idx, 6)  = eij_contact(2)   ! eij_y
	chi_buffer(buffer_idx, 7)  = eij_contact(3)   ! eij_z
	chi_buffer(buffer_idx, 8)  = vrelf_unit(1)   ! ghat_post_x
	chi_buffer(buffer_idx, 9)  = vrelf_unit(2)   ! ghat_post_y
	chi_buffer(buffer_idx, 10) = vrelf_unit(3)   ! ghat_post_z

	ef_buffer(buffer_idx, 1) = Et_00
	ef_buffer(buffer_idx, 2) = Er10
	ef_buffer(buffer_idx, 3) = Er20
	ef_buffer(buffer_idx, 4) = Ef(1)
	ef_buffer(buffer_idx, 5) = Er1_prime
	ef_buffer(buffer_idx, 6) = Er2_prime
	ef_buffer(buffer_idx, 7) = b_contact

	econs_buffer(buffer_idx) = SUM(Ef)/E0
	nphit_buffer(buffer_idx) = NPHIT
	prerot_buffer(buffer_idx, 1) = sqrt(DOT_PRODUCT(VREL0,VREL0))
	prerot_buffer(buffer_idx, 2) = sqrt(DOT_PRODUCT(VRELF,VRELF))

	ftr_buffer(buffer_ftr_idx, 1) = f_tr
	ftr_buffer(buffer_ftr_idx, 2) = delta_E_tr
	ftr_buffer(buffer_ftr_idx, 3) = delta_E_total

	orient_buffer(buffer_orient_idx,  1) = S2_pair
	orient_buffer(buffer_orient_idx,  2) = S2_1n
	orient_buffer(buffer_orient_idx,  3) = S2_2n
	orient_buffer(buffer_orient_idx,  4) = S2_1v
	orient_buffer(buffer_orient_idx,  5) = S2_2v
	orient_buffer(buffer_orient_idx,  6) = cos_u1_n
	orient_buffer(buffer_orient_idx,  7) = cos_u2_n
	orient_buffer(buffer_orient_idx,  8) = cos_u1_v
	orient_buffer(buffer_orient_idx,  9) = cos_u2_v
	orient_buffer(buffer_orient_idx, 10) = u1u2_dot
	orient_buffer(buffer_orient_idx, 11) = contact_lambda
	orient_buffer(buffer_orient_idx, 12) = contact_mu
	orient_buffer(buffer_orient_idx, 13) = E_n_pre
	orient_buffer(buffer_orient_idx, 14) = b_out

	uvec_buffer(buffer_uvec_idx,  1:3)  = U1_pre
	uvec_buffer(buffer_uvec_idx,  4:6)  = U2_pre
	uvec_buffer(buffer_uvec_idx,  7:9)  = U1_post
	uvec_buffer(buffer_uvec_idx, 10:12) = U2_post
	END IF

	IF (WRITE_V2) THEN
		omega1_post = OMEGA(1,1)*U(1,:) + OMEGA(1,2)*UX(1,:) + OMEGA(1,3)*UY(1,:)
		omega2_post = OMEGA(2,1)*U(2,:) + OMEGA(2,2)*UX(2,:) + OMEGA(2,3)*UY(2,:)
		outcome_values = 0.D0
		outcome_values(1:3) = TRY_C1; outcome_values(4:6) = TRY_C2
		outcome_values(7:9) = TRY_W1; outcome_values(10:12) = TRY_W2
		outcome_values(13:15) = TRY_U1; outcome_values(16:18) = TRY_U2
		outcome_values(19:21) = VEL(1,:); outcome_values(22:24) = VEL(2,:)
		outcome_values(25:27) = omega1_post; outcome_values(28:30) = omega2_post
		outcome_values(31:33) = U(1,:); outcome_values(34:36) = U(2,:)
		outcome_values(37:39) = TRY_IMPACT
		outcome_values(40:42) = contact_normal
		outcome_values(43:45) = eij_contact
		outcome_values(46) = contact_lambda/MAX(LCYL + DIA, 1.D-30)
		outcome_values(47) = contact_mu/MAX(LCYL + DIA, 1.D-30)
		outcome_values(48) = Et_f_elastic; outcome_values(49) = Er_f_elastic
		outcome_values(50) = Ef(1); outcome_values(51) = Er1_prime
		outcome_values(52) = Er2_prime; outcome_values(53) = E0
		outcome_values(54) = delta_E_tr; outcome_values(55) = delta_E_rot
		outcome_values(56) = delta_E_total; outcome_values(57) = elastic_rel_error
		outcome_values(58) = b_contact/MAX(LCYL + DIA, 1.D-30)
		outcome_values(59) = b_out
		outcome_values(60:62) = vrel0_unit
		outcome_values(63:65) = vrelf_unit
		CALL BUFFER_OUTCOME(event_id, try_index, NPHIT, outcome_values)
	END IF

	! Flush if buffer full
	IF (WRITE_LEGACY .AND. buffer_idx >= MAX_BUFFER) THEN
		CALL FLUSH_BUFFERS()
	END IF
	return
	end subroutine measure_dem
