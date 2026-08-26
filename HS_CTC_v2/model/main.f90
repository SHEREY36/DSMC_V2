
	program main

	use run_param
	use particles
	use output
	use constants
	use rng_mod
!$ use omp_lib

	implicit none
	DOUBLE PRECISION, DIMENSION(3) :: R12, E12
	DOUBLE PRECISION :: D12, RV12
	INTEGER :: num_threads, event_id, attempt_number
	LOGICAL :: accepted

	! Saved pre-collision state for elastic replay (PRIVATE per thread)
	DOUBLE PRECISION :: POS_SAVE(2,3), VEL_SAVE(2,3), OMEGA_SAVE(2,3)
	DOUBLE PRECISION :: U_SAVE(2,3), UX_SAVE(2,3), UY_SAVE(2,3)

	write(*,*) 'Reading and initializing'
	call read_input()
	call INITIALIZE()

	! Report thread count
	num_threads = 1
!$ num_threads = omp_get_max_threads()
	write(*,*) 'Using ', num_threads, ' thread(s)'

	write(*,*) 'Beginning collisions'

!$OMP PARALLEL PRIVATE(R12, E12, D12, RV12, accepted, attempt_number, &
!$OMP&    POS_SAVE, VEL_SAVE, OMEGA_SAVE,               &
!$OMP&    U_SAVE, UX_SAVE, UY_SAVE)                     &
!$OMP SHARED(NTRY, NHIT, NSAMPLES)
		buffer_idx        = 0
		buffer_ftr_idx    = 0
		buffer_orient_idx = 0
		buffer_uvec_idx   = 0
		attempt_buffer_idx = 0
		outcome_buffer_idx = 0
		ELASTIC_PASS      = .FALSE.

	! Each requested event owns a deterministic random stream. Consequently the
	! same event_id has the same pre-collision state for every alpha and the
	! result is independent of OpenMP scheduling.
!$OMP DO SCHEDULE(STATIC)
	DO event_id = 1, NSAMPLES
		CALL INIT_THREAD_RNG(event_id - 1, RUN_SEED)
		accepted = .FALSE.
		attempt_number = 0
		DO WHILE(.NOT.accepted)
		attempt_number = attempt_number + 1
		CALL INIT_PART()

		! Safety initialisation: if elastic pass misses, f_tr uses Et_00
		Et_f_elastic = Et_00
		Er_f_elastic = Er_00

		! Save pre-collision particle state for inelastic replay
		POS_SAVE = POS;   VEL_SAVE = VEL;   OMEGA_SAVE = OMEGA
		U_SAVE   = U;     UX_SAVE  = UX;    UY_SAVE    = UY

		! ---- ELASTIC PASS (first) ----
		ELASTIC_PASS = .TRUE.
		DO WHILE(.TRUE.)
			PREV_CONTACT = CONTACT
			CALL INTEGRATE_EOM
			IF(.NOT.PREV_CONTACT.AND.CONTACT) NPHIT = NPHIT + 1

			R12 = POS(2,:) - POS(1,:)
			D12 = SQRT(DOT_PRODUCT(R12,R12))
			E12 = VEL(2,:) - VEL(1,:)
			E12 = E12/SQRT(DOT_PRODUCT(E12,E12))
			RV12 = DOT_PRODUCT(E12,R12)

			IF(RV12.GE.0.D0.AND.D12.GT.BMAX&
				.AND.SUM(F).LT.SMALL_NUM.AND.&
				SUM(TAU).LT.SMALL_NUM) EXIT
		END DO
		! Stores Et_f_elastic / Er_f_elastic and returns early
		IF(HIT) CALL MEASURE_DEM(event_id, attempt_number)
		ELASTIC_PASS = .FALSE.

		! Restore particle state for inelastic pass
		POS = POS_SAVE;   VEL = VEL_SAVE;   OMEGA = OMEGA_SAVE
		U   = U_SAVE;     UX  = UX_SAVE;    UY    = UY_SAVE
		CONTACT = .FALSE.; PREV_CONTACT = .FALSE.
		NPHIT = 0;         HIT = .FALSE.

		! ---- INELASTIC PASS (second) ----
		DO WHILE(.TRUE.)
			PREV_CONTACT = CONTACT
			CALL INTEGRATE_EOM
			IF(.NOT.PREV_CONTACT.AND.CONTACT) NPHIT = NPHIT + 1

			R12 = POS(2,:) - POS(1,:)
			D12 = SQRT(DOT_PRODUCT(R12,R12))
			E12 = VEL(2,:) - VEL(1,:)
			E12 = E12/SQRT(DOT_PRODUCT(E12,E12))
			RV12 = DOT_PRODUCT(E12,R12)

			IF(RV12.GE.0.D0.AND.D12.GT.BMAX&
				.AND.SUM(F).LT.SMALL_NUM.AND.&
				SUM(TAU).LT.SMALL_NUM) EXIT
		END DO
		! Computes f_tr, b_out, orient/uvec descriptors and buffers all outputs
		IF (WRITE_V2) CALL BUFFER_ATTEMPT(event_id, attempt_number, HIT)
		IF(HIT) THEN
			CALL MEASURE_DEM(event_id, attempt_number)
			accepted = .TRUE.
!$OMP ATOMIC UPDATE
			NHIT = NHIT + 1
!$OMP END ATOMIC
		END IF

!$OMP ATOMIC UPDATE
		NTRY = NTRY + 1
!$OMP END ATOMIC
		END DO
	END DO
!$OMP END DO
	! Each thread flushes its own remaining buffered data before the region ends
	CALL FLUSH_BUFFERS()
!$OMP END PARALLEL

	write(*,*) 'Number of hits: ', NHIT


	write(*,*) 'Recording Final State'
	call measure_final()

	return
	end program main
