"""Data-only screening, source mapping and physical-setting inheritance."""
from dataclasses import replace
import jax
import numpy as np
import pandas as pd
import pytest
from jaxstar.specfit import (LineSelectionCriteria, Observation, SpecFit, SpecModel, select_lines,
                            single_star_params)
from jaxstar.specfit.line_selection import _SelectedIP
from test_specfit import case

@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def test_data_only_selection_reasons_and_observed_frame():
    wave = 5000*np.exp(np.linspace(-60,60,601)/299792.458)
    v = np.linspace(-60,60,601)
    flux = 1-.4*np.exp(-.5*((v-12)/2)**2)
    obs = Observation(wave,flux,np.full_like(flux,.01))
    lines = pd.DataFrame({'line_id':['target','outside'],'wavelength':[5000.,6000.], 'source':['Doyle','G9']})
    table = select_lines(obs,lines,rv=12.)
    assert table.selected.tolist()==[True,False]
    assert table.source.tolist()==['Doyle','G9']
    assert table.iloc[1].rejected_by=='outside_observed_orders'
    selected=table.iloc[0]
    assert selected.wavelength_min > 5000
    assert selected.wavelength_min < wave[np.argmin(flux)] < selected.wavelength_max
    too_deep=select_lines(obs,lines,rv=12.,criteria={'center_depth_max':.2})
    assert 'depth' in too_deep.iloc[0].rejected_by
    blocked=replace(obs,mask=np.abs(v-12)<4)
    rejected=select_lines(blocked,lines,rv=12.)
    assert not rejected.iloc[0].selected
    with pytest.raises(TypeError):
        LineSelectionCriteria(unknown_rule=1)


def test_selected_ip_keeps_order_and_repeated_sources():
    source=np.stack([np.linspace(4990,5010,100),np.linspace(5990,6010,100)])
    def ip(wave):
        assert wave.shape[0]==2
        assert (wave[0]<5100).all() and (wave[1]>5900).all()
        return dict(mu1_kms=np.stack([.1*(wave[0]-5000),.2*(wave[1]-6000)]), mu2_kms=0., sigma1_kms=1., sigma2_kms=1., fraction2=0.)
    target=np.stack([np.linspace(5999,6001,21),np.linspace(4998,5000,21),np.linspace(6002,6003,21)])
    mapped=_SelectedIP(ip,source,[1,0,1])(target)['mu1_kms']
    np.testing.assert_allclose(mapped, np.stack([.2*(target[0]-6000),.1*(target[1]-5000),.2*(target[2]-6000)]))


def test_selected_specfit_inherits_ip_macro_masks_and_model(monkeypatch):
    sf,_,_=case()
    def ip(wave):
        return dict(mu1_kms=np.zeros_like(wave),mu2_kms=np.zeros_like(wave),
                    sigma1_kms=1+.00005*(wave-5000),sigma2_kms=np.ones_like(wave),fraction2=np.zeros_like(wave))
    parent=SpecModel(sf.specmodel.spectra,fixed_ip=ip,macro_model='gaussian',vmax=30.,teff_interp='linear')
    parent=jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(parent)))
    sf=SpecFit(sf.observation,parent)
    fit_mask=np.zeros(sf.observation.shape,bool);fit_mask[:,48]=True;sf.set_fit_mask(fit_mask)
    options=dict(rv=12.,criteria={'center_prominence_min':.02},wav_margin=1.)
    # Selection and cropping must not evaluate a stellar spectrum.
    with monkeypatch.context() as patch:
        patch.setattr(SpecModel,'__call__',lambda *a,**k: (_ for _ in ()).throw(AssertionError('synthesized during selection')))
        line_sf,table=sf.select_lines([5000.,6000.],**options)
    assert table.selected.all()
    assert line_sf.specmodel.macro_model=='gaussian' and line_sf.specmodel.teff_interp=='linear'
    assert line_sf.specmodel.vmax==30.
    assert line_sf.region_mapping.source_regions==(8,9)
    assert not line_sf.fit_mask.any()
    np.testing.assert_array_equal(sf.fit_mask,fit_mask)
    inherited,_=sf.select_lines([5000.,6000.],inherit_fit_mask=True,**options)
    np.testing.assert_array_equal(inherited.fit_mask,line_sf.region_mapping.take(fit_mask,fill_value=True))
    expected=ip(np.asarray(line_sf.specmodel.spectra.wavelength))
    for name,value in expected.items():
        np.testing.assert_allclose(line_sf.specmodel.fixed_ip[name],value)
    params=single_star_params({'teff':5800.},vsini=6.,vmacro=3.,q1=.36,q2=.3,rv=12.)
    np.testing.assert_allclose(parent(params,line_sf.observation.wavelength),line_sf.specmodel(params,line_sf.observation.wavelength),atol=2e-10)


def test_nan_storage_padding_does_not_split_real_line():
    wave = 5000*np.exp(np.linspace(-30,30,301)/299792.458)
    flux = 1-.4*np.exp(-.5*(np.linspace(-30,30,301)/2)**2)
    original = Observation(wave,flux,np.full_like(flux,.01))
    i = 151
    padded = Observation(np.insert(wave,i,(wave[i-1]+wave[i])/2),
                         np.insert(flux,i,np.nan),np.insert(np.full_like(flux,.01),i,np.nan),
                         mask=np.arange(len(wave)+1)==i)
    first = select_lines(original,[5000.],rv=0.)
    second = select_lines(padded,[5000.],rv=0.)
    assert first.selected[0] and second.selected[0]
    np.testing.assert_allclose(first.center_prominence,second.center_prominence)
