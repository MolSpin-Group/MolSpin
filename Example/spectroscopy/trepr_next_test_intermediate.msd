SpinSystem system1
{
  Spin FE1
  {
    type = electron;
    spin = 1/2;
    tensor = matrix(" 2.0033 0.000 0.000 ; 0.0000 2.0025 0.000 ; 0.000 0.000 2.0021 ");
  }

  Spin WE2
  {
    type = electron;
    spin = 1/2;
    tensor = matrix(" 2.0066 0.000 0.000 ; 0.0000 2.0054 0.000 ; 0.000 0.000 2.0022 ");
  }

  Interaction zeeman1
  {
    type = zeeman;
    field = "0.0 0.0 3.380";
    orientation = -0.3194,-2.0822, -0.4014;
    group1 = FE1;
    ignoretensors = false;
    CommonPrefactor = true;
    Prefactor = 1.0;
  }

  Interaction zeeman2
  {
    type = zeeman;
    field = "0.0 0.0 3.380";
    group1 = WE2;
    ignoretensors = false;
    CommonPrefactor = true;
    Prefactor = 1.0;
  }

  Interaction dipolar
  {
    type = doublespin;
    group1 = FE1;
    group2 = WE2;
    orientation = -1.1956, -1.2497, 0.0000;
    tensor = matrix("0.000082666 0.000000000 0.000000000 ; 0.000000000 0.000082666 0.000000000 ; 0.000000000 0.000000000 -0.000165333");
    ignoretensors = true;
    CommonPrefactor = true;
    prefactor = 2.002319304;
  }

  State Singlet
  {
    spins(FE1,WE2) = |1/2,-1/2> - |-1/2,1/2>;
  }
}

Settings
{
  Settings general
  {
    steps = 240;
    notifications = details;
  }

  Action field_strength1
  {
    type = AddVector;
    vector = system1.zeeman1.field;
    direction = "0 0 1"; value = 0.00005;
  }

  Action field_strength2
  {
    type = AddVector;
    vector = system1.zeeman2.field;
    direction = "0 0 1"; value = 0.00005;
  }
}

Run
{
  Task TrEPR_intermediate
  {
    type = statichs-trepr-spectra;
    mwfrequency = 95.0;
    linewidth_fad = 0.004692082111;
    linewidth_donor = 0.004692082111;
    lineshape = gaussian;
    electron1 = FE1;
    electron2 = WE2;
    fieldinteraction = zeeman2;
    initialstate = Singlet;
    HamiltonianH0list = zeeman1,zeeman2,dipolar;
    powdersamplingpoints = 2000;
    powdergridtype = fibonacci;
    powdergammapoints = 1;
    powderfullsphere = true;
    fulltensorrotation = true;
    sweepcache = false;
    datafile = "trepr_next_test_intermediate.dat";
  }
}
