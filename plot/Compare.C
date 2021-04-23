void Compare(){
  gStyle->SetOptStat(000000);
  TString inputDir  = "/home/xingty/COMET/Others/GAN/draw";
  TString outputDir = "/home/xingty/COMET/Others/GAN/draw";
  TString PID ="2212";
  //Which value do you want to draw?
  //TString name ="posy";
  //double max = 120;
  //double min =-120;
  //Number of bins
  //int nbin = 200;

  //Which value do you want to draw?
  TString name ="time";
  double max =10000;
  double min =300;
  //Number of bins
  int nbin = 200;

  //Time cut
  double timecutlow  = (TMath::Log(TMath::Log(300))   - 1.741)/0.479;
  double timecuthigh = (TMath::Log(TMath::Log(10000)) - 1.741)/0.479;
  TCut cut1=Form("time>%f&&time<%f",300.,10000.);
  TCut cut2=Form("time>%f&&time<%f",timecutlow,timecuthigh);

  //Input root and output name
  TString resampling = inputDir+"/MC4o_Resampling_"+PID+"_Z_4175.root";
  TString MC4o       = inputDir+"/MC4o_"+PID+"_Z_4175.root";
  TString Gan        = inputDir+"/"+PID+".root";
  TString figure     = outputDir+"/Compare_"+PID+"_"+name+".gif";

  TCanvas* c1 = new TCanvas ("c1","Mom",800,600);
  c1->SetLogy();

  TFile* f1 = new TFile(MC4o);
  TFile* f2 = new TFile(resampling);
  TFile* f3 = new TFile(Gan);
  TTree* t_MC4   = (TTree*) f1->Get("Particle");
  TTree* t_Resam = (TTree*) f2->Get("Particle");
  TTree* t_GAN   = (TTree*) f3->Get("ttree");

  TString h1name = "MC4o_"+PID+"_"+name;
  TString h2name = "resam_"+PID+"_"+name;
  TString h3name = "gan_"+PID+"_"+name;
  TH1D* h1= new TH1D(h1name,"MC4o "+name,nbin,min,max);
  TH1D* h2= new TH1D(h2name,"resam "+name,nbin,min,max);
  TH1D* h3= new TH1D(h3name,"gan "+name,nbin,min,max);

  //Draw MC4 and Resampling
  //Without cut
  //int n1=t_MC4->Draw(name+">>"+h1name,"","goff");
  //int n2=t_Resam->Draw(name+">>"+h2name,"","goff");
  //With cut
  int n1=t_MC4->Draw(name+">>"+h1name,cut1,"goff");
  int n2=t_Resam->Draw(name+">>"+h2name,cut1,"goff");

  //Draw GAN, you need to change name of value each time
  //int n3=t_GAN->Draw("(-120 + y*240)>>"+h3name,"","goff");
  //int n3=t_GAN->Draw("(-120 + y*240)>>"+h3name,cut2,"goff");
  //int n3=t_GAN->Draw("(5.34e-05 + theta*2.7499466)>>"+h3name,cut2,"goff");
  //int n3=t_GAN->Draw("(-3.142 + phi*6.284)>>"+h3name,cut2,"goff");
  int n3=t_GAN->Draw("TMath::Exp(TMath::Exp(1.741 + time*0.479))>>"+h3name,cut2,"goff");
  //int n3=t_GAN->Draw("TMath::Exp(-1.088 + momentum*9.278)>>"+h3name,cut2,"goff");

  double scale2to1 = 1.0*n1/n2;
  double scale3to1 = 1.0*n1/n3;

  h1->SetLineColor(kBlue);
  h2->SetLineColor(kRed);
  h2->Scale(scale2to1);
  h3->Scale(scale3to1);
  h3->SetLineColor(kGreen);
  h2->GetXaxis()->SetTitle(name+" (ns)");
  h2->Draw("hist");
  h1->Draw("same,hist");
  h3->Draw("same,hist");

  TLegend* leg1 = new TLegend(0.9,0.9,0.7,0.7);
  leg1->SetHeader("300<time<10000");
  leg1->AddEntry(h1,"MC4o","f");
  leg1->AddEntry(h2,"Resampling","f");
  leg1->AddEntry(h3,"GAN","f");
  leg1->Draw("Same");

  //Calculate chisq
  double diff_MC4_GAN = 0.00000001;
  double diff_MC4_RSP = 0.00000001;
  double diff_MC4_MC4 = 0.00000001;
  diff_MC4_MC4 = h1->Chi2Test(h1,"CHI2/NDF");
  diff_MC4_GAN = h1->Chi2Test(h3,"CHI2/NDF");
  diff_MC4_RSP = h1->Chi2Test(h2,"CHI2/NDF");
  std::cout<<"Diff of MC4 vs MC4: "<<diff_MC4_MC4<<std::endl;
  std::cout<<"Diff of MC4 vs GAN: "<<diff_MC4_GAN<<std::endl;
  std::cout<<"Diff of MC4 vs Resampling: "<<diff_MC4_RSP<<std::endl;

  TLegend* leg2 = new TLegend(0.7,0.9,0.45,0.7);
  leg2->SetHeader("Chi2/ndf of MC4 vs:");
  leg2->AddEntry(h1,Form("MC4           : %.3f",diff_MC4_MC4),"f");
  leg2->AddEntry(h3,Form("GAN           : %.3f",diff_MC4_GAN),"f");
  leg2->AddEntry(h2,Form("Resampling: %.3f",diff_MC4_RSP),"f");
  leg2->Draw("Same");


  c1->SaveAs(figure,"recreate");
  return;
}

