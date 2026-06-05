    SpinSystem system1
    {
      // ---------------------------------------------------------
      // Spins
      // ---------------------------------------------------------

        Spin E1
        {
                type = electron;
                spin = 1/2;
                tensor = isotropic(1);                                  // or tensor = isotropic(-2.0023);  g-factor of 2 is default if not specified
        }

        Spin E2
         {
                type = electron;
                spin = 1/2;
                tensor = isotropic(1);
        }

        Spin H1
        {
                spin = 1/2;
		tensor = isotropic(1);
        }

        Spin H2
        {
                spin = 1/2;
		tensor = isotropic(1);
        }


      // -------------------------
      // Zeeman interaction
      // -------------------------
        Interaction zeeman1
        {
                type = zeeman;
                field = "0.0 0.0 14.1";
                group1 = E1;
                ignoretensors=true;                                     // false if using an electron g-factor
		CommonPrefactor = false;                                // false if using an electron g-factor
		Prefactor = -176.085;                                   //  electron gyromagnetic ratio
        }

        Interaction zeeman2
        {
                type = zeeman;
                field = "0.0 0.0 14.1";
                spins = E2;
                ignoretensors=true;
		CommonPrefactor = false;
		Prefactor =  -176.085;
        }

        Interaction zeeman3H
        {
                type = zeeman;
                field = "0.0 0.0 14.1";
                spins = H1,H2;
                ignoretensors=true;
		CommonPrefactor = false;
		Prefactor = 0.267522;                                   // hydrogen gyromagnetic ratio
        }

      // -------------------------
      // Hyperfine interactions
      // -------------------------
        Interaction electron1hyperfinehydrogenH1
        {
                type = hyperfine;
                group1 = E1;
                group2 = H1;
                ignoretensors = true;
                tensor = isotropic(-2.15772006e-05);                    // testa units
                prefactor = 2.0023;                                     // change intaraction units to rad/ns the 
        }

        Interaction electron1hyperfinehydrogenH2
        {
                type = hyperfine;
                group1 = E1;
                group2 = H2;
                ignoretensors = true;
                tensor = matrix(" -2.14915627e-05 -1.01231219e-05 -2.15772006e-05 ; -1.01231219e-05 -6.25478337e-05 -1.36985072e-05 ;  -2.15772006e-05 -1.36985072e-05 3.52935491e-05 ");
                prefactor = 2.0023;
        }

         Interaction exchange
         {
                type = exchange;
                group1 = E1;
                group2 = E2;
                tensor=isotropic("-0.00017");
                ignoretensors=true;
 	        prefactor = 2.0023;
         }

      // -------------------------
      // Spin States
      // -------------------------

        State Singlet	// |S>
        {
                spins(E1,E2) = |1/2,-1/2> - |-1/2,1/2>;
        }

        State T0	// |T0>
        {
                spins(E1,E2) = |1/2,-1/2> + |-1/2,1/2>;
        }

        State Tp	// |T+>
        {
                spin(E2) = |1/2>;
                spin(E1) = |1/2>;
        }

        State Tm	// |T->
        {
                spin(E2) = |-1/2>;
                spin(E1) = |-1/2>;
        }

        State Identity
        {
        }

        // ---------------------------------------------------------
        // Transitions
        // ---------------------------------------------------------
        Transition Product1
        {
                type = sink;
                source = Singlet;
                rate = 0.001;
        }

        Transition Product2
        {
                type = sink;
                source = Identity;
                rate = 0.001;
        }


       // -------------------------
       // Spin system properties
       // -------------------------

        Properties properties
        {
                // example of using thermal initial state//
                //initialstate = thermal;                                 // Singlet/T0/Tp/Tm/thermal/
                //temperature = 293.0;                                    //specify if thermal initial state 

                // example of using weighted initial state//
                initialstate = T0,Tp,Tm;
                weights = 0.4,0.3,0.3;
        }

        Pulse pulse1
        {
                type = InstantPulse;
                angle = 90;                                             //angle of rotation
                rotationaxis = "1 0 0";                                 // axis of rotation
                group = E1,E2,H1,H2;                                    // spins which are affected by the pulse
        }

        Pulse pulse2
        {
            type = LongPulseStaticField;                                // gamma * B * Spin 
            field = "0.0 0.0 14.1";                                     // pulse static field vector
            pulsetime = 24.0;                                           // time of pulse durations [ns]
            group = E1,E2,H1;                                           // or spins
            prefactorlist = -176.085,-176.085,0.267522;                 // list of gyromagnetic ratios, specify if not using electron g-factor
            commonprefactorlist = false,false,false;                    // true for electrons if using a g-factor
            ignoretensorslist = true,true,true;                         // true if g-matrix tensors should be ignored
            timestep = 0.2;                                             // propagation step in time for the pulse object
        }

        Pulse pulse3
        {
                type = LongPulse; // gamma * B * Spin * cos (omega * t)
                field = "1.0 0.0 0.0";                                  // vector of field amplitudes 
                frequency = 0.00000395;                                 // frequency of mw field in [rad/ns] 
                pulsetime = 100.0; 
                group = E1;
                commonprefactorlist = true;
                ignoretensorslist = false;
                timestep = 0.5;   
        }

        //need to remmove this after testing
        PulseSequence seq
        {
             tau1 = 10;
             tau2 = 20;
             sequence = [pulse1, tau1], [pulse2, tau2], [pulse3, tau1];   
        }

    }
// -------------------------------------------------------------
    Settings
    {
        Settings general
        {
                steps = 1;
                notifications = details;
        }

        // ---------------------------------------------------------
        // Outputs objects
        // ---------------------------------------------------------

        Output orientation
        {
                type = vectorangle;
                vector = system1.zeeman2.field;
                reference = "0 0 1";                                    // Get angle relative to z-axis
        }

        Output orientation
        {
                type = xyz;
                vector = system1.zeeman2.field;
        }

    }

// -------------------------------------------------------------
    Run
    {
        Task Method1
        {
                type = StaticSS-Spectra;
                method = timeevo;                                                        // timeinf or timeevo, timeinf allows to propagate spin density directly to t = inf.
                integration = false;                                                     // specify if method = timeevo, if true allows to integrate yields in time from [0 to T] where T lies in [0, totaltime] interval with the step = timestep
                timestep = 0.1;                                                          // specify if method = timeevo [ns]
                totaltime = 100.0;                                                       // specify if method = timeevo [ns]
                pulsesequence = ["pulse1 10.0"], ["pulse2 15.0"], ["pulse3 20.0"];       // adds pulsesequence to the density matrix before its evolution in time, ["(name of the pulse) (free-evolution time after the pulse)"]   
                cidsp = true;                                                            // cidsp=true allows to calsulate chemically-induced spin polarization
                spinlist = H1,H2;                                                        // list of the spins to calculate polarization for
                logfile = "test.log";
                datafile = "test.dat";
        }
    }