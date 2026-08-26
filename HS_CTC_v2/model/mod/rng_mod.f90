	module rng_mod

	use, intrinsic :: iso_fortran_env, only: int64

	implicit none

	integer(int64), parameter :: RNG_MODULUS = 2147483647_int64
	integer(int64), parameter :: RNG_MULTIPLIER = 16807_int64
	integer(int64), parameter :: RNG_QUOTIENT = 127773_int64
	integer(int64), parameter :: RNG_REMAINDER = 2836_int64
	integer(int64), parameter :: RNG_BASE_SEED = 12345_int64
	integer(int64), parameter :: RNG_THREAD_STRIDE = 104729_int64

	integer(int64) :: rng_state = 1_int64
	logical :: rng_initialized = .false.
	logical :: normal_has_spare = .false.
	double precision :: normal_spare = 0.D0

!$OMP THREADPRIVATE(rng_state, rng_initialized, normal_has_spare, normal_spare)

	contains

	subroutine init_thread_rng(thread_id, base_seed)
		integer, intent(in) :: thread_id
		integer(int64), intent(in), optional :: base_seed
		integer(int64) :: seed_value
		integer(int64) :: selected_seed
		integer :: i

		selected_seed = RNG_BASE_SEED
		if (present(base_seed)) selected_seed = base_seed
		seed_value = modulo(selected_seed + RNG_THREAD_STRIDE * int(thread_id, int64), &
			RNG_MODULUS - 1_int64) + 1_int64
		rng_state = seed_value

		! Warm up the stream so adjacent thread seeds diverge further.
		do i = 1, 8
			call advance_rng_state()
		end do

		rng_initialized = .true.
		normal_has_spare = .false.
	end subroutine init_thread_rng

	double precision function rng_uniform()
		if (.not. rng_initialized) then
			write(*,*) 'Thread RNG used before initialization'
			stop 1
		end if

		call advance_rng_state()
		rng_uniform = dble(rng_state) / dble(RNG_MODULUS)
	end function rng_uniform

	double precision function rng_normal()
		double precision :: radius, angle, u1, u2

		if (normal_has_spare) then
			rng_normal = normal_spare
			normal_has_spare = .false.
			return
		end if

		u1 = max(rng_uniform(), tiny(1.D0))
		u2 = rng_uniform()
		radius = sqrt(-2.D0 * log(u1))
		angle = 8.D0 * atan(1.D0) * u2
		rng_normal = radius * cos(angle)
		normal_spare = radius * sin(angle)
		normal_has_spare = .true.
	end function rng_normal

	subroutine random_rotation_matrix(rotation)
		double precision, intent(out) :: rotation(3,3)
		double precision :: q0, q1, q2, q3, qnorm

		! A normalized four-dimensional Gaussian is uniform on S^3 and
		! therefore gives a Haar-uniform rotation quaternion.
		q0 = rng_normal()
		q1 = rng_normal()
		q2 = rng_normal()
		q3 = rng_normal()
		qnorm = sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
		q0 = q0/qnorm; q1 = q1/qnorm; q2 = q2/qnorm; q3 = q3/qnorm

		rotation(1,1) = 1.D0 - 2.D0*(q2*q2 + q3*q3)
		rotation(1,2) = 2.D0*(q1*q2 - q0*q3)
		rotation(1,3) = 2.D0*(q1*q3 + q0*q2)
		rotation(2,1) = 2.D0*(q1*q2 + q0*q3)
		rotation(2,2) = 1.D0 - 2.D0*(q1*q1 + q3*q3)
		rotation(2,3) = 2.D0*(q2*q3 - q0*q1)
		rotation(3,1) = 2.D0*(q1*q3 - q0*q2)
		rotation(3,2) = 2.D0*(q2*q3 + q0*q1)
		rotation(3,3) = 1.D0 - 2.D0*(q1*q1 + q2*q2)
	end subroutine random_rotation_matrix

	subroutine advance_rng_state()
		integer(int64) :: hi, lo, test_value

		hi = rng_state / RNG_QUOTIENT
		lo = modulo(rng_state, RNG_QUOTIENT)
		test_value = RNG_MULTIPLIER * lo - RNG_REMAINDER * hi

		if (test_value > 0_int64) then
			rng_state = test_value
		else
			rng_state = test_value + RNG_MODULUS
		end if
	end subroutine advance_rng_state

	end module rng_mod
