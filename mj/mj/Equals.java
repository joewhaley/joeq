// Equals.java, created Sun Feb  8 16:38:30 PST 2004 by gback
// Copyright (C) 2003 Godmar Back <gback@stanford.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
import Compil3r.Analysis.IPA.*;
import Util.Collections.*;
import Bootstrap.PrimordialClassLoader;
import java.util.*;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

public class Equals {
    PAProxy r;
    PAResults res;
    public Equals(PAProxy r, PAResults res) {
        this.r = r;
        this.res = res;
    }

    /** 
     * List all callsites where the user calls a.equals(b) on things that can't be equal
     * because the points-to set of a and b are disjoint and a doesn't implement equals().
     */
    public TypedBDD listApplesAndOranges(boolean trace) {
        jq_NameAndDesc equals_nd = new jq_NameAndDesc("equals", "(Ljava/lang/Object;)Z");
        jq_Method equals_m = PrimordialClassLoader.getJavaLangObject().getDeclaredInstanceMethod(equals_nd);
        BDD n_bdd = r.N.ithVar(res.getNameIndex(equals_m));
        TypedBDD esites = (TypedBDD)r.mI.restrict(n_bdd).exist(r.Mset);         // I
        BDD haseq = res.typesThatOverrideEquals();          // T2
        Iterator it = esites.iterator();
        TypedBDD rc = (TypedBDD)r.bdd.zero();
        BDDPairing V2toV1 = r.bdd.makePair(r.V2, r.V1);
        while (it.hasNext()) {
            BDD esite = (BDD)it.next();
            BDD a = r.actual.restrict(esite).relprod(r.Z.ithVar(0), r.Zset);     // V2
            a.replaceWith(V2toV1);
            BDD apt = r.vP.relprod(a, r.V1set);
            BDD aptwt = apt.and(r.hT);                  // H1xH1cxT2
            apt.free();

            aptwt.applyWith(haseq.and(r.H1set), BDDFactory.diff);
            apt = aptwt.exist(r.T2set);
            aptwt.free();

            BDD b = r.actual.restrict(esite).relprod(r.Z.ithVar(1), r.Zset);     // V2
            b.replaceWith(V2toV1);
            BDD bpt = r.vP.relprod(b, r.V1set);         // H1xH1c

            if (apt.and(bpt).isZero()) {
                rc.orWith(esite);
            }
            apt.free();
            bpt.free();
        }
        return rc;
    }

    public String toString() {
        return "Equals - find calls to a.equals(b) where pts(a) and pts(b) is empty";
    }
}
