#2nd implementation of the luminosity calculation with beam beam but variables are hard coded
import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfieldsdevlumi as xf
import xpart as xp
import pickle

# Generating sequences

context = xo.ContextCpu()

p0c = 6500e9
bunch_intensity = 2.5e11#0.7825E11
physemit_x = (2.946E-6*xp.PROTON_MASS_EV)/p0c 
physemit_y = (2.946E-6*xp.PROTON_MASS_EV)/p0c 
beta_x = 19.17
beta_y = 19.17
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 62.31
Qy = 60.32
Qs = 2.1E-3
frev = 11245.5 
nTurn = 1000

n_macroparticles = int(1e4)
shifts = [0, 3, 6]
xs_b1 = []
ys_b1 = []
xs_b2 = []
ys_b2 = []
pxs_b1 = []
pys_b1 = []
pxs_b2 = []
pys_b2 = []


for i in range(len(shifts)):
    pipeline_manager = xt.PipelineManager()
    pipeline_manager.add_particles('b1',0)
    pipeline_manager.add_particles('b2',0)
    pipeline_manager.add_element('IP1')

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

    print('build bb elements...')
    bbeam_b1 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles_b2.q0,
                phi = 0,alpha=0,
                config_for_update = config_for_update_b1,
                ref_shift_x = shifts[i]*np.sqrt(physemit_x*beta_x)/2, 
                flag_numerical_luminosity=1)
    bbeam_b2 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles_b1.q0,
                phi = 0,alpha=0,
                config_for_update = config_for_update_b2,
                ref_shift_x = shifts[i]*np.sqrt(physemit_x*beta_x)/2)


    #################################################################
    # arcs (here they are all the same with half the phase advance) #
    #################################################################

    arc = xt.LineSegmentMap(
            betx = beta_x,bety = beta_y,
            qx = Qx, qy = Qy,bets = beta_s, qs=Qs)
   #################################################################
    # Tracker                                                       #
    #################################################################

    elements_b1 = [bbeam_b1,arc]
    elements_b2 = [bbeam_b2,arc]
    element_names_b1 = ['bbeam_b1','arc']
    element_names_b2 = ['bbeam_b2','arc']
    line_b1 = xt.Line(elements=elements_b1, element_names=element_names_b1)
    line_b2 = xt.Line(elements=elements_b2, element_names=element_names_b2)
    line_b1.build_tracker()
    line_b2.build_tracker()
    branch_b1 = xt.PipelineBranch(line_b1,particles_b1)
    branch_b2 = xt.PipelineBranch(line_b2,particles_b2)
    multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

    #################################################################
    # Tracking                                                      #
    #################################################################

    multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)