
	module run_param
	use, intrinsic :: iso_fortran_env, only: int64
	implicit none
	
	INTEGER :: NTRY, NSAMPLES
	integer(int64), parameter :: EVENT_ID_STRIDE = 10000000_int64
	double precision :: TCOLL, dt
	! dt is chosen per encounter (energy-bound rule), so every thread owns its copy
	!$OMP THREADPRIVATE(dt)
	integer(int64) :: RUN_SEED = 12345_int64
	integer :: ENSEMBLE_ID = 0
	logical :: REPLAY_MODE = .FALSE.
	character(len=512) :: REPLAY_FILE = ''
	integer(int64) :: REPLAY_RECORD_COUNT = 0_int64
	double precision :: TTR_INPUT = 1.D0, TROT_INPUT = 1.D0, AR_INPUT = 1.D0
	character(len=16) :: OUTPUT_MODE = 'v2'
	logical :: WRITE_LEGACY = .FALSE., WRITE_V2 = .TRUE.
	! Conservative advancement of the force-free approach. The pre-contact
	! flight carries no forces, so a large step is exact rather than
	! approximate, and it is 99.996 per cent of the integration steps.
	logical :: FAST_APPROACH = .TRUE.
	! Steps per contact time. The frozen v1 value is 50.
	double precision :: DT_DIVISOR = 50.D0
	! Contact model, set by the REQUIRED environment variables
	!   CTC_DAMP_VELOCITY = contact | center      normal damping velocity
	!   CTC_FORCE_LAW     = hertz   | linear      F = KN DN^1.5 + CN DN^0.25 VRN | KN DN + CN VRN
	!   CTC_DT_RULE       = energy_bound | linear_tc
	! Model C1 = contact, hertz, energy_bound; model R1 (v1) = center, linear, linear_tc.
	character(len=16) :: DAMP_VELOCITY = '', FORCE_LAW = '', DT_RULE = ''
	logical :: DAMP_CONTACT = .FALSE., HERTZ_LAW = .FALSE., DT_ENERGY_BOUND = .FALSE.
	character(len=16) :: CONTACT_MODEL_ID = ''
	! inverse of the smallest effective normal mass of a contact (lever arm L/2 on both rods)
	double precision :: OMEFF_MIN
	! step guard per trajectory pass (a pass that needs more is a bug: the run stops)
	integer(int64) :: MAX_STEPS_PASS = 200000000_int64
	
	contains
	
	end module run_param
