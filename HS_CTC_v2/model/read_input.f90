
	subroutine read_input

	use particles
	use run_param
	use output
	use, intrinsic :: iso_fortran_env, only: int64

	implicit none

	integer :: lunit = 10
	character(len=255) :: filename_input
	double precision :: utemp, wtemp, AR_val
	integer :: num_args
	character(len=100) :: arg_val
	character(len=16) :: mode_arg

	! Check for command-line arguments
	! Usage: SphCyl <alpha> <kTm> <kTI> <AR> [output_dir] [seed] [nsamples] [output_mode]
	! output_mode is v2 (default), legacy, or both. closure is a v1 alias for v2.
	num_args = COMMAND_ARGUMENT_COUNT()

	IF (num_args >= 4) THEN
		! Command-line mode: read ALPHA_PP, kTm, kTI, AR from arguments
		write(*,*) 'Reading parameters from command line'

		! Read system_input.dat for geometry and material properties
		filename_input = 'system_input.dat'
		OPEN(unit=lunit, FILE=filename_input, status='old')
		READ(lunit,*) NSAMPLES
		READ(lunit,*) DIA, LCYL, MASS   ! LCYL will be overridden by AR
		READ(lunit,*) EYoung, GPoisson
		close(lunit)

		! Override parameters from command line
		CALL GET_COMMAND_ARGUMENT(1, arg_val)
		READ(arg_val, *) ALPHA_PP

		CALL GET_COMMAND_ARGUMENT(2, arg_val)
		READ(arg_val, *) utemp
		kTm = utemp
		TTR_INPUT = utemp

		CALL GET_COMMAND_ARGUMENT(3, arg_val)
		READ(arg_val, *) wtemp
		kTI = wtemp
		TROT_INPUT = wtemp

		! Compute LCYL from aspect ratio AR
		! AR = (LCYL + DIA) / DIA  =>  LCYL = (AR - 1) * DIA
		CALL GET_COMMAND_ARGUMENT(4, arg_val)
		READ(arg_val, *) AR_val
		LCYL = (AR_val - 1.0D0) * DIA
		AR_INPUT = AR_val

		ToE = 'T'  ! Force temperature mode for command-line

		! Optional 5th argument: output directory
		IF (num_args >= 5) THEN
			CALL GET_COMMAND_ARGUMENT(5, output_dir)
		ELSE
			output_dir = './'
		END IF

		IF (num_args >= 6) THEN
			CALL GET_COMMAND_ARGUMENT(6, arg_val)
			READ(arg_val, *) RUN_SEED
		END IF

		IF (num_args >= 7) THEN
			CALL GET_COMMAND_ARGUMENT(7, arg_val)
			READ(arg_val, *) NSAMPLES
		END IF

		IF (num_args >= 8) THEN
			CALL GET_COMMAND_ARGUMENT(8, mode_arg)
			OUTPUT_MODE = ADJUSTL(mode_arg)
		ELSE
			OUTPUT_MODE = 'v2'
		END IF

		SELECT CASE (TRIM(OUTPUT_MODE))
		CASE ('legacy')
			WRITE_LEGACY = .TRUE.; WRITE_V2 = .FALSE.
		CASE ('v2', 'closure')
			WRITE_LEGACY = .FALSE.; WRITE_V2 = .TRUE.
		CASE ('both')
			WRITE_LEGACY = .TRUE.; WRITE_V2 = .TRUE.
		CASE DEFAULT
			write(*,*) 'Invalid output_mode; expected v2, legacy, or both'
			stop 2
		END SELECT

		IF (ALPHA_PP <= 0.D0 .OR. ALPHA_PP > 1.D0) THEN
			write(*,*) 'alpha must satisfy 0 < alpha <= 1'
			stop 2
		END IF
		IF (TTR_INPUT <= 0.D0 .OR. TROT_INPUT <= 0.D0) THEN
			write(*,*) 'kTm and kTI must both be positive'
			stop 2
		END IF
		IF (AR_INPUT < 1.D0) THEN
			write(*,*) 'aspect ratio must be at least 1'
			stop 2
		END IF
		IF (NSAMPLES <= 0) THEN
			write(*,*) 'nsamples must be positive'
			stop 2
		END IF
		IF (NSAMPLES >= EVENT_ID_STRIDE) THEN
			write(*,*) 'nsamples must be less than ', EVENT_ID_STRIDE
			stop 2
		END IF
		IF (RUN_SEED < 0_int64 .OR. RUN_SEED > 900000000_int64) THEN
			write(*,*) 'seed must be between 0 and 900000000'
			stop 2
		END IF

		write(*,*) 'ALPHA_PP = ', ALPHA_PP
		write(*,*) 'kTm = ', kTm
		write(*,*) 'kTI = ', kTI
		write(*,*) 'AR   = ', AR_val, '  (LCYL = ', LCYL, ')'
		write(*,*) 'Output dir: ', TRIM(output_dir)
		write(*,*) 'Seed = ', RUN_SEED, '  NSAMPLES = ', NSAMPLES
		write(*,*) 'Output mode: ', TRIM(OUTPUT_MODE)

	ELSE
		! Legacy mode: read all from system_input.dat
		write(*,*) 'Reading parameters from system_input.dat'
		output_dir = './'

		filename_input	= 'system_input.dat'
		OPEN(unit=lunit, FILE=filename_input, status='old')
		READ(lunit,*) NSAMPLES
		READ(LUNIT,*) DIA, LCYL, MASS
		READ(lunit,*) EYoung, GPoisson
		READ(lunit,*) ALPHA_PP
		READ(lunit,*) ToE, utemp, wtemp
		IF(ToE.EQ.'E') THEN
			Ek = utemp;	Er = wtemp
		ELSE
			! Reading in kT (not kT/m) - will be adjusted in initialize
			kTm = utemp; kTI = wtemp
			TTR_INPUT = utemp; TROT_INPUT = wtemp
		END IF
		AR_INPUT = (LCYL + DIA)/DIA
		WRITE_LEGACY = .TRUE.; WRITE_V2 = .TRUE.; OUTPUT_MODE = 'both'
		close(lunit)
	END IF

	return
	end subroutine read_input
