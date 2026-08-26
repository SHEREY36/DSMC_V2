
	module run_param
	use, intrinsic :: iso_fortran_env, only: int64
	implicit none
	
	INTEGER :: NTRY, NSAMPLES
	integer(int64), parameter :: EVENT_ID_STRIDE = 10000000_int64
	double precision :: TCOLL, dt
	integer(int64) :: RUN_SEED = 12345_int64
	double precision :: TTR_INPUT = 1.D0, TROT_INPUT = 1.D0, AR_INPUT = 1.D0
	character(len=16) :: OUTPUT_MODE = 'v2'
	logical :: WRITE_LEGACY = .FALSE., WRITE_V2 = .TRUE.
	
	contains
	
	end module run_param
