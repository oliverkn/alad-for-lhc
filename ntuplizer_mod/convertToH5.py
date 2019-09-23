import numpy as np
import h5py
import ROOT as rt
import math

#####

#rt.gSystem.Load("/afs/cern.ch/user/m/mpierini/work/DataScience/Delphes-3.3.2/libDelphes")
#rt.gInterpreter.Declare('#include "/afs/cern.ch/user/m/mpierini/work/DataScience/Delphes-3.3.2/classes/DelphesClasses.h"')
#rt.gInterpreter.Declare('#include "/afs/cern.ch/user/m/mpierini/work/DataScience/Delphes-3.3.2/external/ExRootAnalysis/ExRootTreeReader.h"')

rt.gSystem.Load("/afs/cern.ch/user/m/mpierini/work/DataScience/Delphes-3.4.1/libDelphes")
rt.gInterpreter.Declare('#include "/afs/cern.ch/user/m/mpierini/work/DataScience/Delphes-3.4.1/classes/DelphesClasses.h"')
rt.gInterpreter.Declare('#include "/afs/cern.ch/user/m/mpierini/work/DataScience/Delphes-3.4.1/external/ExRootAnalysis/ExRootTreeReader.h"')

#####

def PFIso(p, DR, PtMap, subtractPt):
    if p.Pt() <= 0.: return 0.
    DeltaEta = PtMap[:,0] - p.Eta()
    DeltaPhi = PtMap[:,1] - p.Phi()
    pi = rt.TMath.Pi()
    DeltaPhi = DeltaPhi - 2*pi*(DeltaPhi >  pi) + 2*pi*(DeltaPhi < -1.*pi)
    isInCone = DeltaPhi*DeltaPhi + DeltaEta*DeltaEta < DR*DR
    Iso = PtMap[isInCone, 2].sum()/p.Pt()
    if subtractPt: Iso = Iso -1
    return Iso

#####

def ChPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowTrack:
        if h.PT<= 0.5: continue
        pTmap.append([h.Eta, h.Phi, h.PT])
        #nParticles += 1
    #pTmap = np.reshape(pTmap, (nParticles, 3))
    return np.asarray(pTmap)

def NeuPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowNeutralHadron:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
        #nParticles += 1
    #pTmap = np.reshape(pTmap, (nParticles, 3))
    return np.asarray(pTmap)

def PhotonPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowPhoton:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
        #nParticles += 1
    #pTmap = np.reshape(pTmap, (nParticles, 3))
    return np.asarray(pTmap)

#####

def selection(event, TrkPtMap, NeuPtMap, PhotonPtMap, evtID):
    # one electron or muon with pT> 15 GeV
    if event.Electron_size == 0 and event.MuonTight_size == 0: return False, False, False
    foundMuon = None #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0, 0, 1, 1]
    foundEle =  None #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0, 1, 0, 1]
    l = rt.TLorentzVector()
    for ele in event.Electron:        
        if ele.PT <= 23.: continue
        #if ele.PT <= 25.: continue
        l.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
        pfisoCh = PFIso(l, 0.3, TrkPtMap, True) 
        pfisoNeu = PFIso(l, 0.3, NeuPtMap, False) 
        pfisoGamma = PFIso(l, 0.3, PhotonPtMap, False) 
        if foundEle == None and (pfisoCh+pfisoNeu+pfisoGamma)<0.45: 
            #foundEle.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
            foundEle = [evtID, l.E(), l.Px(), l.Py(), l.Pz(), l.Pt(), l.Eta(), l.Phi(), 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 0, 1, 0, ele.Charge]
    for muon in event.MuonTight:
        #if muon.PT <= 23.: continue
        if muon.PT <= 25.: continue
        l.SetPtEtaPhiM(muon.PT, muon.Eta, muon.Phi, 0.)
        pfisoCh = PFIso(l, 0.3, TrkPtMap, True)
        pfisoNeu = PFIso(l, 0.3, NeuPtMap, False)
        pfisoGamma = PFIso(l, 0.3, PhotonPtMap, False)
        if foundMuon == None and (pfisoCh+pfisoNeu+pfisoGamma)<0.45: 
            #foundMuon.SetPtEtaPhiM(muon.PT, muon.Eta, muon.Phi, 0.)
            foundMuon = [evtID, l.E(), l.Px(), l.Py(), l.Pz(), l.Pt(), l.Eta(), l.Phi(), 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 0, 0, 1, muon.Charge]
    if foundEle != None and foundMuon != None:
        if foundEle[5] > foundMuon[5]: return True, foundEle, foundMuon
        else: return True, foundMuon, foundEle
    if foundEle != None: return True, foundEle, foundMuon
    if foundMuon != None: return True, foundMuon, foundEle
    return False, None, None

#####

def Convert(filename, outFileName, hlfonly):
    nTrkLength = 900
    nNeuLength = 400
    nPhotonLength = 300

    inFile = rt.TFile.Open(filename)
    tree = inFile.Get("Delphes")
    q = rt.TLorentzVector()
    # particles are stored as pT, eta, phi, E, pdgID
    particles = []
    # ['HT', 'MET', 'PhiMET', 'MT', 'nJets', 'nBJets','LepPt', 'LepEta', 'LepPhi', 'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge', 'LepIsEle']
    HLF = []
    Nevt = 0
    Nwritten = 0
    print(tree)
    myLep = rt.TLorentzVector()
    NtrkMax = 0
    NphMax = 0
    NneuMax = 0
    NeleMax = 0
    NmuMax = 0
    for event in tree:
        electrons = []
        muons = []
        #if Nevt > 100: continue
        #if Nevt > 500: continue
        # isolation maps
        TrkPtMap = ChPtMapp(0.3, event)
        NeuPtMap = NeuPtMapp(0.3, event)
        PhotonPtMap = PhotonPtMapp(0.3, event)
        if TrkPtMap.shape[0] == 0: continue
        if NeuPtMap.shape[0] == 0: continue
        if PhotonPtMap.shape[0] == 0: continue
        selected, lep, otherlep = selection(event, TrkPtMap, NeuPtMap, PhotonPtMap, Nwritten)
        if not selected: 
            Nevt += 1
            continue
        particles.append(lep)
        lepMomentum = rt.TLorentzVector()
        lepMomentum.SetPtEtaPhiM(lep[5], lep[6], lep[7], 0.)
        # electrons
        for h in event.Electron:
            if len(electrons)>=10: continue
            if h.PT<=0.5: continue
            q.SetPtEtaPhiM(h.PT, h.Eta, h.Phi, 0.)            
            if lepMomentum.DeltaR(q) > 0.0001:
                pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
                pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
                pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
                electrons.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.PT, h.Eta, h.Phi, 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 0, 1, 0, h.Charge])
        # muons 
        for h in event.MuonTight:
            if len(muons)>=10: continue
            if h.PT<=0.5: continue
            q.SetPtEtaPhiM(h.PT, h.Eta, h.Phi, 0.)            
            if lepMomentum.DeltaR(q) > 0.0001:
                pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
                pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
                pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
                muons.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.PT, h.Eta, h.Phi, 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 0, 0, 1, h.Charge])
        # tracks
        nTrk = 0
        for h in event.EFlowTrack:
            if nTrk>=nTrkLength: continue
            if h.PT<=0.5: continue
            q.SetPtEtaPhiM(h.PT, h.Eta, h.Phi, 0.)
            thisIsAnElectron = False
            for ele in electrons:
                myLep.SetPtEtaPhiE(ele[5], ele[6], ele[7], ele[1])
                if myLep.DeltaR(q) <= 0.0001: 
                    # set the X,Y,Z of the electron to the track one
                    ele[8] = h.X
                    ele[9] = h.Y
                    ele[10] = h.Z
                    thisIsAnElectron = True
            thisIsAMuon = False
            for mu in muons:
                myLep.SetPtEtaPhiE(mu[5], mu[6], mu[7], mu[1])
                if myLep.DeltaR(q) <= 0.0001: 
                    # set the X,Y,Z of the electron to the track one
                    mu[8] = h.X
                    mu[9] = h.Y
                    mu[10] = h.Z
                    thisIsAMuon = True
            if thisIsAMuon: continue
            if thisIsAnElectron: continue
            if lepMomentum.DeltaR(q)  <= 0.0001:
                # set the X,Y,Z of the lepton to the track one                                                                       
                lep[8] = h.X
                lep[9] = h.Y
                lep[10] = h.Z
            else:
                pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
                pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
                pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
                particles.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.PT, h.Eta, h.Phi, h.X, h.Y, h.Z, pfisoCh, pfisoGamma, pfisoNeu, 1, 0, 0, 0, 0, np.sign(h.PID)])
                nTrk += 1
        nPhoton = 0
        for h in event.EFlowPhoton:
            if nPhoton >= nPhotonLength: continue
            if h.ET <= 1.: continue
            q.SetPtEtaPhiM(h.ET, h.Eta, h.Phi, 0.)
            pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
            pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
            pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
            particles.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.ET, h.Eta, h.Phi, 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 1, 0, 0, 0])
            nPhoton += 1
        nNeu = 0
        for h in event.EFlowNeutralHadron:
            if nNeu >= nNeuLength: continue
            if h.ET <= 1.: continue
            q.SetPtEtaPhiM(h.ET, h.Eta, h.Phi, 0.)
            pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
            pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
            pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False) 
            particles.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.ET, h.Eta, h.Phi, 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 1, 0, 0, 0, 0])
            nNeu += 1
        if nTrk>NtrkMax: NtrkMax = nTrk
        if nNeu>NneuMax: NneuMax = nNeu
        if len(electrons)>NeleMax: NeleMax = len(electrons)
        if len(muons)>NmuMax: NmuMax = len(muons)
        if nPhoton>NphMax: NphMax = nPhoton
        # append leptons
        for ele in electrons: particles.append(ele)
        for iEle in range(len(electrons),10):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for mu in muons: particles.append(mu)
        for iMuon in range(len(muons),10):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for iTrk in range(nTrk, nTrkLength):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for iPhoton in range(nPhoton, nPhotonLength):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for iNeu in range(nNeu, nNeuLength):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # HIGH-LEVEL FEATURES
        myMET = event.MissingET[0]
        MET = myMET.MET
        METv = rt.TLorentzVector()
        METv.SetPtEtaPhiM(MET, 0., myMET.Phi, 0.)
        phiMET = METv.DeltaPhi(lepMomentum)
        MT = math.sqrt(2.*MET*lepMomentum.Pt()*(1-rt.TMath.Cos(lepMomentum.Phi()-phiMET)))
        # sphericity
        #p = np.array(particles)[:,5:7]
        #SPH = np.linalg.norm(np.sum(p, axis=0))/np.sum(np.linalg.norm(p, axis=1))
        # HT and (b)jet multiplicity
        HT = 0
        nJets = 0
        nBjets = 0
        sumJet = rt.TLorentzVector()
        for jet in event.Jet:
            if jet.PT > 30. and abs(jet.Eta)<2.6: 
                nJets +=1
                HT += jet.PT
                if jet.BTag >0: nBjets += 1
                thisJet = rt.TLorentzVector()
                thisJet.SetPtEtaPhiM(jet.PT, jet.Eta, jet.Phi, jet.Mass)
                sumJet = sumJet + thisJet
        massJet = sumJet.M()
        LepPt = lep[5]
        LepEta = lep[6]
        LepPhi = lep[7]
        LepIsoCh = lep[11]
        LepIsoGamma = lep[12]
        LepIsoNeu = lep[13]
        LepCharge = lep[19]
        LepIsEle = lep[17]
        nMuToWrite = len(muons)
        nEleToWrite = len(electrons)
        sumEle = rt.TLorentzVector()
        for ele in electrons:
            sumEle = sumEle + rt.TLorentzVector(ele[2], ele[3], ele[4], ele[1])
        sumMu = rt.TLorentzVector()
        for mu in muons:
            sumMu = sumMu + rt.TLorentzVector(mu[2], mu[3], mu[4], mu[1])
        if LepIsEle: 
            nEleToWrite += 1
            sumEle = sumEle + rt.TLorentzVector(lep[2], lep[3], lep[4], lep[1])
        else: 
            nMuToWrite += 1
            sumMu = sumMu + rt.TLorentzVector(lep[2], lep[3], lep[4], lep[1])
        massEle = sumEle.M()
        massMu = sumMu.M()
        ptEle = sumEle.Pt()
        ptMu = sumMu.Pt()
        HLF.append([Nwritten, HT, MET, phiMET, MT, nJets, nBjets, massJet, LepPt, LepEta, LepIsoCh, LepIsoGamma, LepIsoNeu, LepCharge, LepIsEle, nMuToWrite, massMu, ptMu, nEleToWrite, massEle, ptEle, nTrk, nNeu, nPhoton])
        Nevt += 1
        Nwritten += 1
    ##### NOW SAVE INTO H5
    #HLF = HLF[15:]
    #nRows = int(HLF.shape[0]/15)
    #HLF = HLF.reshape((nRows,15))
    #HLFpandas = pd.DataFrame({'EvtId':HLF[:,0],'HT':HLF[:,1], 'MET':HLF[:,2], 'PhiMET':HLF[:,3], 'MT':HLF[:,4], 'nJets':HLF[:,5], 'bJets':HLF[:,6],\
    #                              'LepPt':HLF[:,7],'LepEta':HLF[:,8],'LepPhi':HLF[:,9],'LepIsoCh':HLF[:,10],'LepIsoGamma':HLF[:,11],'LepIsoNeu':HLF[:,12],\
    #                             'LepCharge':HLF[:,13],'LepIsEle':HLF[:,14]})
    #HLFpandas.to_hdf(filename.replace(".root",".h5"),'HLF')
    if len(HLF) !=0:
        print(outFileName)
        f = h5py.File(outFileName, "w")
        f.create_dataset('HLF', data=np.asarray(HLF), compression='gzip')
        f.create_dataset('HLF_Names', data=np.array(['EvtId','HT', 'MET', 'DPhiMETLep', 'MT', 'nJets', 'bJets', 'allJetMass', 'LepPt', 'LepEta', \
                                                    'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge', 'LepIsEle', 'nMu', 'allMuMass', 'allMuPt', \
                                                         'nEle', 'allEleMass', 'allElePt', 'nChHad', 'nNeuHad', 'nPhoton']), compression='gzip')
        if not hlfonly:
            pArray = np.asarray(particles).reshape((Nwritten,(nNeuLength+nPhotonLength+nTrkLength+1+10+10), 20))
            f.create_dataset('Particles', data=pArray, compression='gzip')
            f.create_dataset('Particles_Names', data=np.array(['EvtId', 'Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 'vtxX', 'vtxY', 'vtxZ', \
                                                                   'ChPFIso', 'GammaPFIso', 'NeuPFIso', 'isChHad', 'isNeuHad', 'isGamma', 'isEle', 'isMu', 'Charge']),\
                                 compression='gzip')
        f.close()
        print("Max Number of Electrons: %i" %NeleMax)
        print("Max Number of Muons:     %i" %NmuMax)
        print("Max Number of Trks:      %i" %NtrkMax)
        print("Max Number of Neu:       %i" %NneuMax)
        print("Max Number of Photons:  %i" %NphMax)

    #PARTICLESPanda = pd.DataFrame({'EvtId':particles[:,0], \
    #                               'Energy': particles[:,1], \
    #                               'Px': particles[:,2], \
    #                               'Py': particles[:,3], \
    #                               'Pz': particles[:,4], \
    #                               'Pt': particles[:,5], \
    #                               'Eta': particles[:,6], \
    #                               'Phi': particles[:,7], \
    #                               'vtxX': particles[:,8], \
    #                               'vtxY': particles[:,9], \
    #                               'vtxZ': particles[:,10], \
    #                               'ChPFIso': particles[:,11], \
    #                               'GammaPFIso': particles[:,12], \
    #                               'NeuPFIso': particles[:,13], \
    #                               'isChHad': particles[:,14], \
    #                               'isNeuHad': particles[:,15], \
    #                               'isGamma': particles[:,16], \
    #                               'isEle': particles[:,17], \
    #                               'isMu': particles[:,18], \
    #                               'Charge': particles[:,19]})
    #PARTICLESPanda.to_hdf(filename.replace(".root",".h5"),'Particles')
    #with open(filename.replace(".root",".csv"), "wb") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(events)

if __name__ == "__main__":
    import sys
    print sys.argv[1]
    inFile = rt.TFile.Open(sys.argv[1])    
    tree = inFile.Get("Delphes")
    print("Nevt: %i" %tree.GetEntries())
    hlfonly = sys.argv[3].lower()=='true'
    Convert(sys.argv[1], sys.argv[2], hlfonly)
    #Convert(sys.argv[1], sys.argv[2], sys.argv[3])
