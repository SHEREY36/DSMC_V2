.SUFFIXES:
.SUFFIXES: .f90 .o

FC = gfortran
FCFLAGS ?= -g -fopenmp -O3
EXECUTABLE = SphCyl

MAINDIR = ../model
MODDIR = ../model/mod

MODOBJS = \
constants_mod.o \
rng_mod.o \
output_mod.o \
particle_mod.o \
run_param_mod.o

OBJS = \
calc_force_dem.o \
initialize.o \
init_part.o \
integrate_eom.o \
measure_dem.o \
measure_final.o \
read_input.o

ALLOBJS = $(OBJS) $(MODOBJS)

.PHONY: all
all: $(EXECUTABLE)

$(EXECUTABLE) : $(MAINDIR)/main.f90 $(ALLOBJS)
	$(FC) $(FCFLAGS) $(ALLOBJS) -o $@ $<

# Modules
###########################################################
particle_mod.o : $(MODDIR)/particle_mod.f90
	$(FC) $(FCFLAGS) -c $<
	
run_param_mod.o : $(MODDIR)/run_param_mod.f90
	$(FC) $(FCFLAGS) -c $<

constants_mod.o : $(MODDIR)/constants_mod.f90
	$(FC) $(FCFLAGS) -c $<
	
output_mod.o : $(MODDIR)/output_mod.f90
	$(FC) $(FCFLAGS) -c $<

rng_mod.o : $(MODDIR)/rng_mod.f90
	$(FC) $(FCFLAGS) -c $<

# Other
###########################################################
initialize.o : $(MAINDIR)/initialize.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<

read_input.o : $(MAINDIR)/read_input.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<

# DES
###########################################################
integrate_eom.o : $(MAINDIR)/integrate_eom.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<

calc_force_dem.o: $(MAINDIR)/calc_force_dem.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<
	
init_part.o : $(MAINDIR)/init_part.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<
	
# Measure
###########################################################
measure_dem.o : $(MAINDIR)/measure_dem.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<
	
measure_final.o : $(MAINDIR)/measure_final.f90 $(MODOBJS)
	$(FC) $(FCFLAGS) -c $<
	
.PHONY : clean

clean :
	rm -f *.mod *.o $(EXECUTABLE)
