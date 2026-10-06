	module replay_mod
	use, intrinsic :: iso_fortran_env, only: int64
	use run_param, only: REPLAY_MODE, REPLAY_FILE, REPLAY_RECORD_COUNT, &
		NSAMPLES, RUN_SEED
	implicit none

	integer, parameter :: REPLAY_WIDTH = 18
	double precision, allocatable :: REPLAY_RECORDS(:,:)

	contains

	subroutine LOAD_REPLAY_DATA()
		integer :: unit, status
		integer(int64) :: byte_count

		if (.not. REPLAY_MODE) return
		inquire(file=trim(REPLAY_FILE), size=byte_count, iostat=status)
		if (status /= 0 .or. byte_count <= 0_int64) then
			write(*,*) 'Replay file is missing or empty: ', trim(REPLAY_FILE)
			stop 4
		end if
		if (mod(byte_count, int(8 * REPLAY_WIDTH, int64)) /= 0_int64) then
			write(*,*) 'Replay file size is not a multiple of one 18-float64 record'
			stop 4
		end if
		REPLAY_RECORD_COUNT = byte_count / int(8 * REPLAY_WIDTH, int64)
		if (REPLAY_RECORD_COUNT < int(NSAMPLES, int64)) then
			write(*,*) 'Replay file has fewer records than requested hit events'
			stop 4
		end if
		allocate(REPLAY_RECORDS(REPLAY_WIDTH, int(REPLAY_RECORD_COUNT)))
		open(newunit=unit, file=trim(REPLAY_FILE), status='old', access='stream', &
			form='unformatted', convert='little_endian', action='read', iostat=status)
		if (status /= 0) then
			write(*,*) 'Could not open replay file: ', trim(REPLAY_FILE)
			stop 4
		end if
		read(unit, iostat=status) REPLAY_RECORDS
		close(unit)
		if (status /= 0) then
			write(*,*) 'Could not read complete replay file: ', trim(REPLAY_FILE)
			stop 4
		end if
		write(*,*) 'Loaded ', REPLAY_RECORD_COUNT, ' replay pair records'
	end subroutine LOAD_REPLAY_DATA

	subroutine GET_REPLAY_RECORD(event_id, attempt_number, record)
		integer, intent(in) :: event_id, attempt_number
		double precision, intent(out) :: record(REPLAY_WIDTH)
		integer(int64) :: offset, index

		if (.not. allocated(REPLAY_RECORDS)) then
			write(*,*) 'Replay data requested before it was loaded'
			stop 4
		end if
		! Use a deterministic two-dimensional lattice in (event,attempt).
		! The former contiguous/NSAMPLES stride reused many records whenever
		! one event missed and a later event hit on its first attempt. These
		! coprime strides decorrelate retries while remaining OpenMP-schedule
		! independent and reproducible.
		offset = modulo(RUN_SEED * 104729_int64, REPLAY_RECORD_COUNT)
		index = 1_int64 + modulo(offset &
			+ int(event_id - 1, int64) * 130363_int64 &
			+ int(attempt_number - 1, int64) * 155921_int64, &
			REPLAY_RECORD_COUNT)
		record = REPLAY_RECORDS(:, int(index))
	end subroutine GET_REPLAY_RECORD

	end module replay_mod
