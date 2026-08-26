	module output

	use, intrinsic :: iso_fortran_env, only: int32, int64, real64
	implicit none

	INTEGER :: NHIT
	LOGICAL :: HIT
	DOUBLE PRECISION :: CSX, PROJ_AREA
	DOUBLE PRECISION, DIMENSION(3) :: VREL0, WREL0
	DOUBLE PRECISION :: E0, Er_00, Et_00, Er_1, Er_2
	DOUBLE PRECISION :: TMEAN, RMEAN, b_impact, b_contact
	DOUBLE PRECISION :: Et_f_elastic, Er_f_elastic
	DOUBLE PRECISION :: contact_lambda, contact_mu, mu_in
	DOUBLE PRECISION, DIMENSION(3) :: eij_contact, contact_normal
	DOUBLE PRECISION, DIMENSION(3) :: C1_pre, C2_pre
	DOUBLE PRECISION, DIMENSION(3) :: OMEGA1_lab_pre, OMEGA2_lab_pre
	DOUBLE PRECISION :: S2_pair, S2_1n, S2_2n, S2_1v, S2_2v
	DOUBLE PRECISION :: cos_u1_n, cos_u2_n, cos_u1_v, cos_u2_v, u1u2_dot
	DOUBLE PRECISION, DIMENSION(3) :: U1_pre, U2_pre, U1_post, U2_post
	DOUBLE PRECISION :: E_n_pre
	LOGICAL :: SIM_CONTINUE
	CHARACTER(LEN=255) :: output_dir

	! Exact incoming proposal state. These values exist for hits and misses.
	DOUBLE PRECISION, DIMENSION(3) :: TRY_C1, TRY_C2, TRY_W1, TRY_W2
	DOUBLE PRECISION, DIMENSION(3) :: TRY_U1, TRY_U2, TRY_IMPACT

	!$OMP THREADPRIVATE(HIT, VREL0, WREL0, PROJ_AREA, E0, Er_00, Et_00,        &
	!$OMP& Er_1, Er_2, TMEAN, RMEAN, b_impact, b_contact,                     &
	!$OMP& Et_f_elastic, Er_f_elastic, contact_lambda, contact_mu, mu_in,     &
	!$OMP& eij_contact, contact_normal, C1_pre, C2_pre,                       &
	!$OMP& OMEGA1_lab_pre, OMEGA2_lab_pre, S2_pair, S2_1n, S2_2n, S2_1v,    &
	!$OMP& S2_2v, cos_u1_n, cos_u2_n, cos_u1_v, cos_u2_v, u1u2_dot,          &
	!$OMP& U1_pre, U2_pre, U1_post, U2_post, E_n_pre,                         &
	!$OMP& TRY_C1, TRY_C2, TRY_W1, TRY_W2, TRY_U1, TRY_U2, TRY_IMPACT)

	LOGICAL :: ELASTIC_PASS
	!$OMP THREADPRIVATE(ELASTIC_PASS)

	INTEGER, PARAMETER :: MAX_BUFFER = 1000
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER, 10) :: chi_buffer
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER, 7) :: ef_buffer
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER) :: econs_buffer
	INTEGER, DIMENSION(MAX_BUFFER) :: nphit_buffer
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER, 2) :: prerot_buffer
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER, 3) :: ftr_buffer
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER, 14) :: orient_buffer
	DOUBLE PRECISION, DIMENSION(MAX_BUFFER, 12) :: uvec_buffer
	INTEGER :: buffer_idx, buffer_ftr_idx, buffer_orient_idx, buffer_uvec_idx
	!$OMP THREADPRIVATE(chi_buffer, ef_buffer, econs_buffer, nphit_buffer,     &
	!$OMP& prerot_buffer, ftr_buffer, orient_buffer, uvec_buffer,             &
	!$OMP& buffer_idx, buffer_ftr_idx, buffer_orient_idx, buffer_uvec_idx)

	! Schema v2.0.0: typed little-endian headers followed by float64 payloads.
	INTEGER, PARAMETER :: N_ATTEMPT_REAL = 21, N_OUTCOME_REAL = 65
	INTEGER(INT64), DIMENSION(MAX_BUFFER) :: attempt_event, attempt_index, attempt_block
	INTEGER(INT32), DIMENSION(MAX_BUFFER) :: attempt_hit
	REAL(REAL64), DIMENSION(MAX_BUFFER, N_ATTEMPT_REAL) :: attempt_real
	INTEGER :: attempt_buffer_idx
	INTEGER(INT64), DIMENSION(MAX_BUFFER) :: outcome_event, outcome_index, outcome_block
	INTEGER(INT32), DIMENSION(MAX_BUFFER) :: outcome_ncontact
	REAL(REAL64), DIMENSION(MAX_BUFFER, N_OUTCOME_REAL) :: outcome_real
	INTEGER :: outcome_buffer_idx
	!$OMP THREADPRIVATE(attempt_event, attempt_index, attempt_block, attempt_hit, &
	!$OMP& attempt_real, attempt_buffer_idx, outcome_event, outcome_index,        &
	!$OMP& outcome_block, outcome_ncontact, outcome_real, outcome_buffer_idx)

	contains

	SUBROUTINE BUFFER_ATTEMPT(event_id, try_index, did_hit)
		INTEGER, INTENT(IN) :: event_id, try_index
		LOGICAL, INTENT(IN) :: did_hit
		attempt_buffer_idx = attempt_buffer_idx + 1
		attempt_event(attempt_buffer_idx) = INT(event_id, INT64)
		attempt_index(attempt_buffer_idx) = INT(try_index, INT64)
		attempt_block(attempt_buffer_idx) = MOD(INT(event_id - 1, INT64), 128_INT64)
		attempt_hit(attempt_buffer_idx) = MERGE(1_INT32, 0_INT32, did_hit)
		attempt_real(attempt_buffer_idx, 1:3) = TRY_C1
		attempt_real(attempt_buffer_idx, 4:6) = TRY_C2
		attempt_real(attempt_buffer_idx, 7:9) = TRY_W1
		attempt_real(attempt_buffer_idx, 10:12) = TRY_W2
		attempt_real(attempt_buffer_idx, 13:15) = TRY_U1
		attempt_real(attempt_buffer_idx, 16:18) = TRY_U2
		attempt_real(attempt_buffer_idx, 19:21) = TRY_IMPACT
		IF (attempt_buffer_idx >= MAX_BUFFER) CALL FLUSH_ATTEMPT_BUFFER()
	END SUBROUTINE BUFFER_ATTEMPT

	SUBROUTINE BUFFER_OUTCOME(event_id, try_index, n_contact, values)
		INTEGER, INTENT(IN) :: event_id, try_index, n_contact
		REAL(REAL64), INTENT(IN) :: values(N_OUTCOME_REAL)
		outcome_buffer_idx = outcome_buffer_idx + 1
		outcome_event(outcome_buffer_idx) = INT(event_id, INT64)
		outcome_index(outcome_buffer_idx) = INT(try_index, INT64)
		outcome_block(outcome_buffer_idx) = MOD(INT(event_id - 1, INT64), 128_INT64)
		outcome_ncontact(outcome_buffer_idx) = INT(n_contact, INT32)
		outcome_real(outcome_buffer_idx,:) = values
		IF (outcome_buffer_idx >= MAX_BUFFER) CALL FLUSH_OUTCOME_BUFFER()
	END SUBROUTINE BUFFER_OUTCOME

	SUBROUTINE FLUSH_ATTEMPT_BUFFER()
		INTEGER :: i
		IF (attempt_buffer_idx <= 0) RETURN
!$OMP CRITICAL(v2_attempt_write)
		DO i = 1, attempt_buffer_idx
			WRITE(1010) attempt_event(i), attempt_index(i), attempt_block(i), &
				attempt_hit(i), 0_INT32, attempt_real(i,:)
		END DO
		FLUSH(1010)
!$OMP END CRITICAL(v2_attempt_write)
		attempt_buffer_idx = 0
	END SUBROUTINE FLUSH_ATTEMPT_BUFFER

	SUBROUTINE FLUSH_OUTCOME_BUFFER()
		INTEGER :: i
		IF (outcome_buffer_idx <= 0) RETURN
!$OMP CRITICAL(v2_outcome_write)
		DO i = 1, outcome_buffer_idx
			WRITE(1012) outcome_event(i), outcome_index(i), outcome_block(i), &
				outcome_ncontact(i), 0_INT32, outcome_real(i,:)
		END DO
		FLUSH(1012)
!$OMP END CRITICAL(v2_outcome_write)
		outcome_buffer_idx = 0
	END SUBROUTINE FLUSH_OUTCOME_BUFFER

	SUBROUTINE FLUSH_BUFFERS()
		INTEGER :: i
		IF (buffer_idx > 0) THEN
!$OMP CRITICAL(file_write)
			DO i = 1, buffer_idx
				WRITE(1001,'(10(E14.8,2X))') chi_buffer(i,:)
				WRITE(1002,'(7(E14.8,2X))') ef_buffer(i,:)
				WRITE(1003,'(E14.8)') econs_buffer(i)
				WRITE(1004,*) nphit_buffer(i)
				WRITE(1111,'(2(E14.8,2X))') prerot_buffer(i,:)
			END DO
!$OMP END CRITICAL(file_write)
			buffer_idx = 0
		END IF
		IF (buffer_ftr_idx > 0) THEN
!$OMP CRITICAL(ftr_write)
			DO i = 1, buffer_ftr_idx
				WRITE(1005,'(3(E14.8,2X))') ftr_buffer(i,:)
			END DO
!$OMP END CRITICAL(ftr_write)
			buffer_ftr_idx = 0
		END IF
		IF (buffer_orient_idx > 0) THEN
!$OMP CRITICAL(orient_write)
			DO i = 1, buffer_orient_idx
				WRITE(1006,'(14(E14.8,2X))') orient_buffer(i,:)
			END DO
!$OMP END CRITICAL(orient_write)
			buffer_orient_idx = 0
		END IF
		IF (buffer_uvec_idx > 0) THEN
!$OMP CRITICAL(uvec_write)
			DO i = 1, buffer_uvec_idx
				WRITE(1007,'(12(E14.8,2X))') uvec_buffer(i,:)
			END DO
!$OMP END CRITICAL(uvec_write)
			buffer_uvec_idx = 0
		END IF
		CALL FLUSH_ATTEMPT_BUFFER()
		CALL FLUSH_OUTCOME_BUFFER()
	END SUBROUTINE FLUSH_BUFFERS

	end module output
