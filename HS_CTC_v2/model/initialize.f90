	subroutine initialize
	
	use particles
	use run_param
	use constants
	use output
	
	implicit none
	INTEGER :: I

! Output Files
!---------------------------------------------------------
	CHARACTER(LEN=300) :: filepath

	! Create output directory if it does not exist
	CALL EXECUTE_COMMAND_LINE('mkdir -p ' // TRIM(output_dir))

	IF (WRITE_LEGACY) THEN
		filepath = TRIM(output_dir) // '/csx.txt'
		open(unit=1000, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/chi.txt'
		open(unit=1001, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/Ef.txt'
		open(unit=1002, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/EnergyCons.txt'
		open(unit=1003, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/NPhit.txt'
		open(unit=1004, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/projArea.txt'
		open(unit=2000, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/init.txt'
		open(unit=100,  status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/PreRotEnergy.txt'
		open(unit=1111, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/ftr_data.txt'
		open(unit=1005, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/orient.dat'
		open(unit=1006, status='replace', file=filepath)

		filepath = TRIM(output_dir) // '/uvec.dat'
		open(unit=1007, status='replace', file=filepath)
	END IF

	IF (WRITE_V2) THEN
		filepath = TRIM(output_dir) // '/attempts_v2.bin'
		open(unit=1010, status='replace', file=filepath, access='stream', &
			form='unformatted', convert='little_endian')
		filepath = TRIM(output_dir) // '/outcomes_v2.bin'
		open(unit=1012, status='replace', file=filepath, access='stream', &
			form='unformatted', convert='little_endian')
	END IF

	!open(unit=2925, status='replace', file='ovito.txt')

! Particle Properties
!---------------------------------------------------------
	! Particle Geometry/Material
	RAD = DIA*0.5D0; hLCYL = LCYL*0.5D0
	DIASQ = DIA**2.D0
	BMAX = LCYL + DIA; BMAX = BMAX*1.01D0
	PVOL = PI*(DIA**3.D0)/6.D0 + PI*LCYL*(RAD**2.D0)
	RHO = MASS/PVOL
	
	! Moment of Inertia
	moI(2) = PI/48.D0*RHO*(DIA**2.D0)*(LCYL**3.D0) +&
		3.D0*PI/64.D0*RHO*(DIA**4.D0)*LCYL +&
		PI/60.D0*RHO*(DIA**5.D0) +&
		PI/24.D0*RHO*(DIA**3.D0)*(LCYL**2.D0)
	moI(3) = moI(2)
	moI(1) = PI/32.D0*RHO*(DIA**4.D0)*LCYL + &
		PI/60.D0*RHO*(DIA**5.D0)
	DO I = 1,3
		omoI(I) = 1.D0/moI(I)
	END DO

	! Hertzian Spring
	EStar = EYoung/(2.D0*(1.D0-(GPoisson**2.D0)))
	KN = 4.D0/3.D0*SQRT(RAD*0.5D0)*EStar
	! Damper
	BETA = -LOG(ALPHA_PP)/(SQRT((PI**2.D0)+(LOG(ALPHA_PP))**2.D0))
	CN = 2.D0*BETA*SQRT(MASS*0.5D0*KN)
	! Use one alpha-independent integration step for common random numbers along
	! an alpha line. The elastic Hertz contact time is also the conservative
	! reference replay time scale; damping only changes the force law.
	TCOLL = PI/SQRT(KN/(MASS*0.5D0))

! Sampling Temperatures
!---------------------------------------------------------
	! STD = kT/m / kT/I
	IF(ToE.EQ.'E') THEN
		SQEk = SQRT(Ek/MASS); SQEr = SQRT(Er*OMOI(2))
	ELSE
		kTm = kTm/MASS; kTI = kTI*OMOI(2)
	END IF

	IF (WRITE_V2) THEN
		filepath = TRIM(output_dir) // '/metadata_v2.json'
		open(unit=1011, status='replace', file=filepath)
		write(1011,'(A)') '{'
		write(1011,'(A)') '  "schema_version": "2.1.0",'
		write(1011,'(A)') '  "byte_order": "little",'
		write(1011,'(A,I0,A)') '  "attempt_record_bytes": ', 32 + 8*N_ATTEMPT_REAL, ','
		write(1011,'(A,I0,A)') '  "outcome_record_bytes": ', 32 + 8*N_OUTCOME_REAL, ','
		write(1011,'(A,ES24.16,A)') '  "alpha": ', ALPHA_PP, ','
		write(1011,'(A,ES24.16,A)') '  "theta": ', TTR_INPUT/TROT_INPUT, ','
		write(1011,'(A,ES24.16,A)') '  "aspect_ratio": ', AR_INPUT, ','
		write(1011,'(A,ES24.16,A)') '  "temperature_translational": ', TTR_INPUT, ','
		write(1011,'(A,ES24.16,A)') '  "temperature_rotational": ', TROT_INPUT, ','
		write(1011,'(A,ES24.16,A)') '  "velocity_scale": ', SQRT(2.D0*kTm), ','
		write(1011,'(A,ES24.16,A)') '  "omega_scale": ', SQRT(2.D0*kTI), ','
		write(1011,'(A,ES24.16,A)') '  "mass": ', MASS, ','
		write(1011,'(A,ES24.16,A)') '  "moi_perpendicular": ', moI(2), ','
		write(1011,'(A,ES24.16,A)') '  "proposal_area": ', 4.D0*BMAX*BMAX, ','
		write(1011,'(A,ES24.16,A)') '  "collision_cross_section": ', &
			PI*DIA*DIA*(0.32D0*AR_INPUT**2.D0 + 0.694D0*AR_INPUT - 0.0213D0), ','
		write(1011,'(A)') '  "collision_cross_section_scale": 1.0,'
		write(1011,'(A)') '  "collision_cross_section_model": "frozen_v1_polynomial",'
		write(1011,'(A,I0,A)') '  "nsamples": ', NSAMPLES, ','
		write(1011,'(A,I0,A)') '  "seed": ', RUN_SEED, ','
		write(1011,'(A)') '  "rng_contract": "event_stream_common_across_alpha",'
		write(1011,'(A,A,A)') '  "output_mode": "', TRIM(OUTPUT_MODE), '",' 
		write(1011,'(A)') '  "normal_contact_velocity": "translational_relative_velocity_only",'
		write(1011,'(A)') '  "finalized": false'
		write(1011,'(A)') '}'
		close(1011)
	END IF

! For Outputs
!---------------------------------------------------------
	NHIT = 0
	NTRY = 0
	TMEAN = 0.D0; RMEAN = 0.D0
	SIM_CONTINUE = .TRUE.
	return
	end subroutine initialize
