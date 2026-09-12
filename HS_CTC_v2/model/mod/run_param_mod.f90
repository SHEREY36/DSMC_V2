
	module run_param
	use, intrinsic :: iso_fortran_env, only: int64
	implicit none
	
	INTEGER :: NTRY, NSAMPLES
	integer(int64), parameter :: EVENT_ID_STRIDE = 10000000_int64
	double precision :: TCOLL, dt
	integer(int64) :: RUN_SEED = 12345_int64
	integer :: ENSEMBLE_ID = 0
	double precision :: TTR_INPUT = 1.D0, TROT_INPUT = 1.D0, AR_INPUT = 1.D0
	character(len=16) :: OUTPUT_MODE = 'v2'
	logical :: WRITE_LEGACY = .FALSE., WRITE_V2 = .TRUE.
	! Conservative advancement of the force-free approach. The pre-contact
	! flight carries no forces, so a large step is exact rather than
	! approximate, and it is 99.996 per cent of the integration steps.
	logical :: FAST_APPROACH = .TRUE.
	! Steps per Hertzian contact time. The frozen v1 value is 50.
	double precision :: DT_DIVISOR = 50.D0
	
	contains
	
	end module run_param
