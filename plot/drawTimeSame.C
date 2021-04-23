{
  //TFile* f1= new TFile("resamp/MC4o_Resampling_-11_Z_4175.root");
  //TFile* f2= new TFile("MC4o/MC4o_-11_Z_4175.root");
  TFile* f1= new TFile("resamp/MC4o_Resampling_2212_Z_4175.root");
  TFile* f2= new TFile("MC4o/MC4o_2212_Z_4175.root");
  TTree* t1=(TTree*) f1->Get("Particle");
  TTree* t2=(TTree*) f2->Get("Particle");
  double min=200;
  double max=10000;
  TH1D* h1= new TH1D("resam","resam",100000,min,max);
  TH1D* h2= new TH1D("MC4o","MC4o",100000,min,max);
  t1->Draw("time>>resam","time>200&&time<1e4","goff");
  t2->Draw("time>>MC4o","time>200&&time<1e4","goff");
  h1->SetLineColor(kRed);
  h1->Draw();
  h2->Draw("same");
}
