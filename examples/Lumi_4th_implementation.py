#Attempting to delete all hardcoding from beambeam3d.py and having variables as inputs instead
import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfieldsdevlumi as xf
import xpart as xp
import time
# Generating sequences

context = xo.ContextCpu()

p0c = 6500e9
bunch_intensity = 0.7825E11
physemit_x = (2.946E-6*xp.PROTON_MASS_EV)/p0c 
physemit_y = (2.946E-6*xp.PROTON_MASS_EV)/p0c 
beta_x = 19.17
beta_y = 19.17
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 64.31
Qy = 59.32
Qs = 2.1E-3
frev = 11245.5 
nTurn = 3

n_macroparticles = int(3)
xs_b1 = []
ys_b1 = []
xs_b2 = []
ys_b2 = []
pxs_b1 = []
pys_b1 = []
pxs_b2 = []
pys_b2 = []

pipeline_manager = xt.PipelineManager()
pipeline_manager.add_particles('b1',0)
pipeline_manager.add_particles('b2',0)
pipeline_manager.add_element('IP1')
pipeline_manager.add_element('IP2')

particles_b1 = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)),
    px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
    y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
    py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
    zeta=sigma_z*np.random.randn(n_macroparticles),
    delta=sigma_delta*np.random.randn(n_macroparticles),
    weight=bunch_intensity/n_macroparticles
)
particles_b1.init_pipeline('b1')
particles_b2 = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)),
    px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
    y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
    py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
    zeta=sigma_z*np.random.randn(n_macroparticles),
    delta=sigma_delta*np.random.randn(n_macroparticles),
    weight=bunch_intensity/n_macroparticles
)
particles_b2.init_pipeline('b2')

#############
# Beam-beam #
#############
slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=1, mode = 'shatilov')
config_for_update_b1 = xf.ConfigForUpdateBeamBeamBiGaussian3D(
pipeline_manager=pipeline_manager,
element_name='IP1',
partner_particles_name = 'b2',
slicer=slicer,
update_every=1,
)
config_for_update_b2 = xf.ConfigForUpdateBeamBeamBiGaussian3D(
pipeline_manager=pipeline_manager,
element_name='IP1',
partner_particles_name = 'b1',
slicer=slicer,
update_every=1,
)

IP2_config_for_update_b1 = xf.ConfigForUpdateBeamBeamBiGaussian3D(
pipeline_manager=pipeline_manager,
element_name='IP2',
partner_particles_name = 'b2',
slicer=slicer,
update_every=1,
)
IP2_config_for_update_b2 = xf.ConfigForUpdateBeamBeamBiGaussian3D(
pipeline_manager=pipeline_manager,
element_name='IP2',
partner_particles_name = 'b1',
slicer=slicer,
update_every=1,
)

print('build bb elements...')
bbeam_b1 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b2.q0,
            phi = 0,alpha=0,
            config_for_update = config_for_update_b1,
            flag_numerical_luminosity = 1,
            n_lumigrid_cells=1500,
            sig_lumigrid_cells=np.sqrt(physemit_x*beta_x),
            range_lumigrid_cells = 24,
            n_macroparticles=n_macroparticles,
            nTurn = nTurn,
            update_lumigrid_sum=0
            )
bbeam_b2 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b1.q0,
            phi = 0,alpha=0,
            config_for_update = config_for_update_b2,
            flag_numerical_luminosity = 1,
            n_lumigrid_cells=1500,
            sig_lumigrid_cells=np.sqrt(physemit_x*beta_x),
            range_lumigrid_cells = 24,
            n_macroparticles=n_macroparticles,
            nTurn = nTurn,
            update_lumigrid_sum=0
            )
print(n_macroparticles)
IP2_bbeam_b1 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b2.q0,
            phi = 0,alpha=0,
            config_for_update = IP2_config_for_update_b1,
            flag_numerical_luminosity = 0,
            update_lumigrid_sum=0)
IP2_bbeam_b2 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b1.q0,
            phi = 0,alpha=0,
            config_for_update = IP2_config_for_update_b2,
            flag_numerical_luminosity=0,
            update_lumigrid_sum=0)


#################################################################
# arcs (here they are all the same with half the phase advance) #
#################################################################

arc = xt.LineSegmentMap(
        betx = beta_x,bety = beta_y,
        qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs)
#################################################################
# Tracker                                                       #
#################################################################
'''
elements_b1 = [bbeam_b1,arc]
elements_b2 = [bbeam_b2,arc]
element_names_b1 = ['bbeam_b1','arc']
element_names_b2 = ['bbeam_b2','arc']
'''
elements_b1 = [bbeam_b1,arc, IP2_bbeam_b1]
elements_b2 = [bbeam_b2,arc, IP2_bbeam_b2]

line_b1 = xt.Line(elements=elements_b1)
line_b2 = xt.Line(elements=elements_b2)
line_b1.build_tracker()
line_b2.build_tracker()
branch_b1 = xt.PipelineBranch(line_b1,particles_b1)
branch_b2 = xt.PipelineBranch(line_b2,particles_b2)


multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

record_qss_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                        capacity={
                                                            "beamstrahlungtable": int(0),
                                                            "bhabhatable": int(0),
                                                            "lumitable": nTurn,
                                                            "numlumitable": nTurn
                                                        })


print('Tracking...')
time0 = time.time()

multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)
print('Done with tracking.',(time.time()-time0)/10,'[s/turn]')
line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

record_qss_b1.move(_context=xo.context_default)

lumi_b1_beambeam = record_qss_b1.lumitable.luminosity
num_lumi_b1_beambeam = bbeam_b1.numlumitable.numerical_luminosity
grid = bbeam_b1.lumigrid_sum
print('Gaussian Luminosity with beam-beam:',frev*lumi_b1_beambeam)
print('Numerical Integrator Luminosity with beam-beam:',num_lumi_b1_beambeam)
print(np.sum(grid))
def Lumi_analytical(Nb, N1, N2, frev, Delta_i, sig_i, sig_x, sig_y):
    W = np.exp(-Delta_i**2/(4*sig_i**2))
    return ((Nb * N1 * N2 * frev * W)/(4 * np.pi * sig_x * sig_y))

print('analytical:',Lumi_analytical(1, bunch_intensity, bunch_intensity, frev, 0,np.sqrt(physemit_x*beta_x), np.sqrt(physemit_x*beta_x), np.sqrt(physemit_y*beta_x)))
