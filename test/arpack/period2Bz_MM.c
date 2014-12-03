#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "A_parametry.h"

const char *nazwa=plikFz;
const char *mode=ModeP;
//************ J **************
double J1xy=jx1; //-1.40;
double J1z=jz1;
double J2xy=jx2;
double J2z=jz2;
//************ D **************
double D1=Dz1;
double D2=Dz2;
//************ g **************
double g1=gz1;
double g2=gz2;//1.983;
//************ B **************
double B0=stalaBg;
double dB=1e-2;
//************ T **************
double T0=T0pod;//2.08515;
double krT=krTpod;
int iloscT=iloscTpod;

int Nn1=(2*s1+1);
int Nn2=(2*s2+1);
int nn1=cn;

double **He1;
double **H1;
double **H2;
float *kwektor;
float *wwektor;
double Tm[cn][cn];
int delta;

double element(double x, int n)
{if(n==1) return x;
else	return element(x/n, n-1);
}

int Hamiltonian(double **Hc, float ss1, float ss2, double jz, double jx, double b1,
		double gg1, double gg2, double d1, double d2, double kk, int n1,int n2)
{int i,n;
 int j,k;
 double **Hp;
 double **He;
 double **Ht;
 double dh;
 double tym,tym1;
 j=n1*n2;

	He = malloc(j * sizeof *He);
	Ht = malloc(j * sizeof *Ht);
	Hp = malloc(j * sizeof *Hp);
	for (i = 0; i<j; ++i) 
		{Ht[i] = malloc(j * sizeof **Ht);
		 Hp[i] = malloc(j * sizeof **Hp);
		 He[i] = malloc(j * sizeof **He);
	    for(n=0;n<j;n++)
			{He[i][n]=0; Hc[i][n]=0;Ht[i][n]=0;Hp[i][n]=0;}

}

n=0;
	for(i=0;i<n1;i++)
		for(j=0;j<n2;j++)
			{dh=ss1-i;tym=ss2-j;
			tym1=(-jz*dh*tym + mgB*b1/2*(gg1*dh+gg2*tym)-d1*dh*dh-d2*tym*tym)/kk;
			 Ht[n][n]=tym1; He[n][n]=tym1; Hc[n][n]=tym1+1;
			 n++;}

	for(i=0;i<n1-1;i++)
		for(j=0;j<n2-1;j++)
			{tym=-jx*sqrt((j+1)*(2*ss2-j)*(i+1)*(2*ss1-i))/(2*kk);
			 n=i*n2+j+1;k=(i+1)*n2+j;
			 Ht[n][k]=tym; Ht[k][n]=tym;
			 He[n][k]=tym; He[k][n]=tym;
			 Hc[n][k]=tym; Hc[k][n]=tym;
			}

int m; n=2;
for(m=0; m<n1; m++) //n1-2+2
	{for(i=0; i<n1*n2; i++)
		{for(j=0; j<n1*n2; j++)
			{Hp[i][j]=0;
			 for(k=0; k<n1*n2; k++)
				Hp[i][j]=Hp[i][j]+He[i][k] * Ht[k][j];
			}
		}
	for(i=0; i<n1*n2; i++)
		for(j=0; j<n1*n2; j++)
			{Ht[i][j]=Hp[i][j]; 
			 Hc[i][j]=Hc[i][j]+element(Hp[i][j],n);
			}
	n++;
	}

if(n1>n2) m=n2;
else m=n1;

dh=1.;

while((dh)>1e-15)
	{for(i=0; i<n1*n2; i++)
		{for(j=0; j<n1*n2; j++)
			{Hp[i][j]=0;
			 for(k=0; k<n1*n2; k++)
			 Hp[i][j] += He[i][k] * Ht[k][j];
			}
		}
	tym=Hc[n2-1][(n2-1)*m];tym1=Hc[0][0];
	for(i=0; i<n1*n2; i++)
		for(j=0; j<n1*n2; j++)
			{Ht[i][j]=Hp[i][j]; 
			 Hc[i][j]=Hc[i][j]+element(Hp[i][j],n);;
			}
    dh=fabs(tym-Hc[n2-1][(n2-1)*m]);
    tym=fabs(tym1-Hc[0][0]);
	if(dh<tym)
	dh=tym;
n++;
}
	for (i = 0; i<n1*n2; ++i) 
		{free(Hp[i]);free(Ht[i]);free(He[i]);}
	free(Hp);free(Ht);free(He);

return n;
}

double Ft(double **Ham1, double **Ham2,float kwektor[],float wwektor[],int Nn, int n1, int n2)
{double suma=0.; int k, s0; float suma1=0.;

for(s0=0;s0<M;s0++)
suma1=suma1+kwektor[2*s0]-kwektor[2*s0+1]-wwektor[2*s0]+wwektor[2*s0+1];

suma1=suma1*10;  //test

if (suma1==0)
{for(s0=0;s0<n2;s0++)
{int m0=0; k=1; float tym1=s2-s0; float w21=tym1;
double sum1=1;
	while(k>0&&m0<M-1)
	{delta=(int)(kwektor[2*m0]-kwektor[2*m0+1]);
	 tym1=w21+delta;
	 if(tym1>-s2-1&&tym1<s2+1)
	 {if(delta>=0)
		 sum1=sum1*Ham2[delta][(int)((s2-w21-delta)*(n1-delta)+s1-kwektor[2*m0])];
	  if(delta<0)
		 sum1=sum1*Ham2[abs(delta)][(int)((s2-tym1+delta)*(n1+delta)+s1-kwektor[2*m0+1])];
	delta=(int)(wwektor[2*m0+1]-wwektor[2*m0+2]);
	  w21=tym1+delta;
	  if(w21>=-s2&&w21<=s2)
	  {if(delta>0)
		 sum1=sum1*Ham1[delta][(int)((s1-wwektor[2*m0+2]-delta)*(n2-delta)+s2-w21)];
	  if(delta<=0)
		 sum1=sum1*Ham1[abs(delta)][(int)((s1-wwektor[2*m0+1]+delta)*(n2+delta)+s2-tym1)];
}
	  else {k=0; sum1=0.;}
}
	 else {k=0; sum1=0.;}
	 m0++;
}
	if(k>0)
	{delta=(int)(kwektor[2*Nn-2]-kwektor[2*Nn-1]);
	 tym1=w21+delta;
	if(tym1>=-s2&&tym1<=s2)
	{if(delta>=0)
		 sum1=sum1*Ham2[delta][(int)((s2-w21-delta)*(n1-delta)+s1-kwektor[2*Nn-2])];
	  if(delta<0)
		 sum1=sum1*Ham2[abs(delta)][(int)((s2-tym1+delta)*(n1+delta)+s1-kwektor[2*Nn-1])];
	delta=(int)(wwektor[2*Nn-1]-wwektor[0]);
	w21=tym1+delta;
	if(w21>=-s2&&w21<=s2)
	{if(delta>0)
		 sum1=sum1*Ham1[delta][(int)((s1-wwektor[0]-delta)*(n2-delta)+s2-w21)];
	  if(delta<=0)
		 sum1=sum1*Ham1[abs(delta)][(int)((s1-wwektor[2*Nn-1]+delta)*(n2+delta)+s2-tym1)];
}
	else sum1=0.;
}
	else sum1=0.;
}
suma=suma+sum1;
}}
return suma;
}

int wiersz(float kwektor[],float wwektor[], int i, int jj,int b,double **Ham1,double **Ham2,int Nn,int n1,int n2)
{int j=jj; int kk;

for(kk=0;kk<n1;kk++)
  {kwektor[2*M-b]=s1-kk;
    if(b==1)
		{Tm[i][j]= Ft(Ham1,Ham2,kwektor,wwektor,Nn,n1,n2); 
j++;}
    else
		j=wiersz(kwektor,wwektor,i,j,b-1,Ham1,Ham2,Nn,n1,n2);
}
return j;
}

int kolumna(float kwektor[],float wwektor[],int ni,int j,int b1,int b,double **Ham1,double **Ham2,int Nn, int n1, int n2)
{int i=ni; int zz;
for(zz=0;zz<n1;zz++)
{wwektor[2*M-b1]=s1-zz;
if(b1==1)
	{wiersz(kwektor,wwektor,i,j,b,Ham1,Ham2,Nn,n1,n2);
i++;
}
else
	i=kolumna(kwektor,wwektor,i,j,b1-1,b,Ham1,Ham2,Nn,n1,n2);
}
return i;
}

int main()
{FILE *f;
 int i,j,n,m,k,p;
 int krokT;
 double wynik=0.;
 double podatnosc;
 double pp[3];
//CuMnM3Tz20-300
f = fopen(nazwa,mode);
if (f == NULL) {perror("Nie udalo sie otworzyc pliku 'wynik.dat' do zapisu");}
else {fprintf(f, "# **********************************************\n");
      fprintf(f, "# *                    Report                  *\n");
      fprintf(f, "# * M    = %d                                   *\n",M); 
      fprintf(f, "# * s1   = %f      ",s1);
      fprintf(f, " s2   = %f      *\n",s2);
      fprintf(f, "# * g1   = %f      ",g1);
      fprintf(f, " g2   = %f      *\n",g2);
      fprintf(f, "# * J1xy = %lf    ",J1xy);
      fprintf(f, "   J1z  = %lf     *\n",J1z);
      fprintf(f, "# * J2xy = %lf    ",J2xy);
      fprintf(f, "   J2z  = %lf     *\n",J2z);
      fprintf(f, "# * D1   = %f      ",D1);
      fprintf(f, " D2   = %f    *\n",D2);
      fprintf(f, "# * T0   = %f                            *\n",T0);
      fprintf(f, "# * Bz   = %lf      ",B0);
      fprintf(f, " dB   = %lf      *\n",dB);
      fprintf(f, "# **********************************************\n");
      fprintf(f, "#T \t\t  podatnosc\n");	


	j=Nn1*Nn2;
	He1 = malloc(j * sizeof *He1);
	for (i = 0; i<j; ++i)
		{He1[i] = malloc(j * sizeof **He1);
		for(n=0;n<j;n++)
			He1[i][n]=0;
}
	Hamiltonian(He1,s1,s2,J1z,J1xy,B0,g1,g2,D1,D2,T0*M,Nn1,Nn2);

	if(Nn1>Nn2) m=Nn2;
	else m=Nn1;

	H1 = malloc(m * sizeof *H1);
	for (n = 0; n<m; ++n) 
		{j=(Nn2-n)*(Nn1-n);
		 H1[n] = malloc(j * sizeof **H1);
		 k=0;
		 for(i=0;i<Nn1-n;i++)
			for(j=0;j<Nn2-n;j++)
				{H1[n][k]=He1[i*Nn2+j+n][(i+n)*Nn2+j];
				 k++;
}
}
	Hamiltonian(He1,s2,s1,J2z,J2xy,B0,g2,g1,D2,D1,T0*M,Nn2,Nn1);

	H2 = malloc(m * sizeof *H2);
	for (n = 0; n<m; ++n) 
		{j=(Nn2-n)*(Nn1-n);
		 H2[n] = malloc(j * sizeof **H2);
		 k=0;
		 for(i=0;i<Nn2-n;i++)
			for(j=0;j<Nn1-n;j++)
				{H2[n][k]=He1[i*Nn1+j+n][(i+n)*Nn1+j];
				 k++;
}
}		
	for (i = 0; i<(Nn1*Nn2); ++i) 
		free(He1[i]);
	free(He1);

	kwektor = malloc(2*M * sizeof *kwektor);
	wwektor = malloc(2*M * sizeof *wwektor);

	for(i=0;i<2*M;i++)
		{kwektor[i]=s1;kwektor[i]=s1;wwektor[i]=s1;wwektor[i]=s1;}

	kolumna(kwektor,wwektor,0,0,2*M,2*M,H1,H2,M,Nn1,Nn2);

	free(kwektor);free(wwektor);

	for (n = 0; n<m; ++n)
	{free(H1[n]);
	 free(H2[n]);
}
	free(H1);free(H2);

		 for(i=0;i<cn;i++) {
			for(j=0;j<cn;j++) {
				fprintf(f, "%f,\t",Tm[i][j]);
				}
			fprintf(f, "\n");
        }
        
      fflush(f);      
      fclose(f);
}

return 0;
}
