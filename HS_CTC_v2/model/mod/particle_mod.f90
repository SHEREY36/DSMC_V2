
	MODULE PARTICLES
	IMPLICIT NONE

	! Particle Geometry
	DOUBLE PRECISION :: DIA, RAD, DIASQ, PVOL
	DOUBLE PRECISION :: LCYL, hLCYL
	! Max Impact Parameter
	DOUBLE PRECISION :: BMAX
	! Particle Density/Mass
	DOUBLE PRECISION :: RHO, MASS
	! Particle Material Prop.
	DOUBLE PRECISION :: EYoung, EStar, GPoisson
	DOUBLE PRECISION :: KN
	! Dissipation
	DOUBLE PRECISION :: ALPHA_PP, BETA, CN 
	! Moment of Inertia
	DOUBLE PRECISION, DIMENSION(3) :: moI(3), omoI(3)
	! Temperatures
	DOUBLE PRECISION :: kTm, kTI
	! Particle Pos/Vel
	DOUBLE PRECISION, DIMENSION(2,3) :: POS, VEL, F
	! Particle Orientation
	DOUBLE PRECISION, DIMENSION(2,3) :: U, UX, UY
	! Angular velocities along each of the particle axes (not the fixed frame)
	DOUBLE PRECISION, DIMENSION(2,3) :: OMEGA, TAU
	! Energy for Translation / Rotation
	! Either Temperatures or energies should be defined
	DOUBLE PRECISION :: Ek, Er, SQEk, SQEr
	CHARACTER(LEN=1) :: ToE
	! Contact?
	LOGICAL :: PREV_CONTACT, CONTACT
	! Number of Particle Hits
	INTEGER :: NPHIT

	! Per-collision state: each thread runs an independent collision
	!$OMP THREADPRIVATE(POS, VEL, F, U, UX, UY, OMEGA, TAU, &
	!$OMP&             PREV_CONTACT, CONTACT, NPHIT)
	CONTAINS

	END MODULE PARTICLES
