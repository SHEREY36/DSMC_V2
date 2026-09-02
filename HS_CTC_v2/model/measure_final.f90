
	subroutine measure_final()

	use output
	use run_param
	use particles
	use constants

	implicit none

	! Flush any remaining buffered data
	CALL FLUSH_BUFFERS()

	IF (WRITE_LEGACY) THEN
		write(1000,'(E14.8)') 4.D0*DBLE(NHIT)/DBLE(NTRY)*BMAX*BMAX
		FLUSH(1000)
		close(1000); close(1001); close(1002); close(1003); close(1004); close(1111)
		close(1005); close(1006); close(1007); close(2000); close(100)
	END IF
	IF (WRITE_V2) THEN
		close(1010); close(1012)
		open(unit=1014, status='replace', file=TRIM(output_dir)//'/_RAW_SUCCESS')
		write(1014,'(A)') 'schema=2.2.0'
		close(1014)
	END IF
	RETURN

	end subroutine measure_final
