// CollectionType.java, created Nov 26, 2003 by gback
// Copyright (C) 2003 Godmar Back <gback@stanford.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Analysis.IPA;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.InputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_Member;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Util.Assert;
import Util.Strings;
import Util.Collections.Pair;
import Util.Collections.HashWorklist;
import Util.Collections.UnmodifiableIterator;
import Util.Collections.GenericMultiMap;
import Util.Graphs.PathNumbering;

/**
 * Do analysis of types stored in collections.
 * 
 * @author Godmar Back
 * @version $Id$
 */
public class CollectionType {

    PAResults res;
    
    public CollectionType(PAResults res) {
        this.res = res;
    }

    public GenericMultiMap cmethods = new GenericMultiMap();
    {	// not static so file is reread everytime you invoke this command 

	// Fileformat see below: type mname mdesc #pidx
        File f = new File("collectionmethods");
	DataInput in = null;
	
        if (f.exists()) {
	    try {
		in = new DataInputStream(new FileInputStream(f));
	    } catch (FileNotFoundException e) {
		e.printStackTrace();
	    }
	} else {
	    InputStream sin = Bootstrap.PrimordialClassLoader.loader.getResourceAsStream("/Support/collectionmethods");
	    if (sin != null) {
		in = new DataInputStream(sin);
	    } else {
		String fContent = 
		    "add 2 Ljava/util/List; add (ILjava/lang/Object;)V\n" +
		    "add 2 Ljava/util/List; set (ILjava/lang/Object;)Ljava/lang/Object;\n" +
		    "add 1 Ljava/util/Collection; add (Ljava/lang/Object;)Z\n" +
		    "add 1 Ljava/util/Vector; addElement (Ljava/lang/Object;)V\n" +
		    "add 2 Ljava/util/Map; put (Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;\n" +
		    "addall 1 Ljava/util/Collection; addAll (Ljava/util/Collection;)Z\n" +
		    "addall 2 Ljava/util/List; addAll (ILjava/util/Collection;)Z\n" +
		    "addall 1 Ljava/util/LinkedList; <init> (Ljava/util/Collection;)V\n";
		in = new DataInputStream(new ByteArrayInputStream(fContent.getBytes()));
	    }
	}

	try {
	    for (;;) {
		String s = in.readLine();
		if (s == null) break;
		if (s.startsWith("#"))
		    continue;
		StringTokenizer st = new StringTokenizer(s);
		String what = st.nextToken();
		String paramidx = st.nextToken();
		jq_Member m = jq_Member.read(st);
		if (m == null) {
		    System.out.println("Could not resolve `" + s + "', ignoring it");
		    continue;
		}
		cmethods.add(new Pair(m.getDeclaringClass(), m.getNameAndDesc()), new CollCall((jq_Method)m, paramidx, what));
	    }
	} catch (IOException e) {
	    e.printStackTrace();
        }
	// System.out.println(cmethods);
    }

    static class CollCall {
	String what;	/* "add", "addall", "consttype", "musthave", "return" */
	String pidx;	/* -1 for return */
	jq_Method meth;

	CollCall(jq_Method meth, String pidx, String what) {
	    this.meth = meth;
	    this.pidx = pidx;
	    this.what = what;
	}
    }

    /**
     * Get all entries that applies to a given method. 
     *
     * @return collection of entries, or null
     */
    private Collection getApplicableEntries(jq_Method m) {
	// see if any superclass or implemented interface declares the method 
	// that is being invoked as a collection-add method
	jq_Class mclass = m.getDeclaringClass();
	while (mclass != null) {
	    Collection values = cmethods.getValues(new Pair(mclass, m.getNameAndDesc()));
	    if (values.size() > 0)
		return values;
	    mclass = mclass.getSuperclass();
	}
	jq_Class []ifs = m.getDeclaringClass().getInterfaces();
	for (int i = 0; i < ifs.length; i++) {
	    Collection values = cmethods.getValues(new Pair(ifs[i], m.getNameAndDesc()));
	    if (values.size() > 0)
		return values;
	}
	return null;
    }

    TypedBDD storedin;  // H1xH2		(Objects x Collection) (+context)
    TypedBDD copiedin;  // H1xH2		(Objects x Collection) (+context)
    TypedBDD musthave;  // V1xH1xH2		(Objects x Collection) (+context)
    TypedBDD supertypes;// H2xH2cxT1 		(Collection x Type)

    public TypedBDD getMustHaves() { return musthave; }

    public BDD getPointsToForP0andPi(PA r, ProgramLocation call, int iidx, CollCall cc, boolean keepV1) {
	int pidx = Integer.parseInt(cc.pidx);
	BDD V1cset = r.V1c.set();
	BDD V1set = r.V1.set();
	BDD isite = r.I.ithVar(iidx);
	BDDPairing V2toV1 = r.bdd.makePair(r.V2, r.V1);

	// ----------- begin of work-around --------------
	/* It turns out that because casts are ignored, the points-to 
	 * information of the actuals might contain entries that can be flatly
	 * ruled out when looking at the declared types of nominal formals of this 
	 * call.  We try to determine those; this is a bit clumsy and doesn't
	 * work for abstract or interface methods.
	 */
	BDD formalFilter = null;		// H1 x Z
	BDD mi = null;
	if (call.isSingleTarget()) {
	    mi = r.IE.restrict(isite);
	} else {
	    BDD b1 = r.mI.restrict(isite);
	    BDD b2 = b1.exist(r.Mset);
	    b1.free();
	    long []nidx = r.N.getVarIndices(b2);
	    b2.free();
	    if (nidx.length > 1)
		System.out.println("warning more than one entry in mI for I("+iidx+")");
	    long m = nidx[0];
	    if (m >= r.Nmap.size()) {
		System.out.println("warning Nmap doesn't have method " + m);
	    } else {
		jq_Method mm = (jq_Method)r.Nmap.get((int)m);
		if (!r.Mmap.contains(mm)) {
		    System.out.println("warning Mmap doesn't have method " + mm);
		} else {
		    mi = r.M.ithVar(res.getMethodIndex(mm));
		}
	    }
	}
	if (mi != null) {
	    BDD formals = r.formal.restrict(mi);
	    if (formals.isZero()) {
		// System.out.print("warning no formals for " + res.toString((TypedBDD)mi, -1));
	    } else {
		mi.free();
		formalFilter = formals.relprod(r.vPfilter, V1set);	// H1 x Z
	    }
	    formals.free();
	} else {
	    System.out.println("warning no suitable methods for formals for " 
		+ call + " I("+iidx+")");
	}
	// ----------- end of work-around --------------
    
	BDD actuals = r.actual.restrict(isite);		// V2xZ
	isite.free();
	BDD z0 = r.Z.ithVar(0);
	BDD v0 = actuals.restrict(z0);			// V2
	v0.replaceWith(V2toV1);				// V1
	BDD v0pt = r.vP.relprod(v0, V1set);			// V1cxH1xH1c
	v0.free(); 
	if (r.NNfilter != null) v0pt.andWith(r.NNfilter.id());
	// begin work-around
	if (formalFilter != null) v0pt.andWith(formalFilter.restrict(z0));
	// end work-around
	z0.free();
	v0pt.replaceWith(r.H1toH2);				// V1cxH2xH2c

	BDD vp = null;
	if (pidx == -1) {
	    vp = r.Iret.restrict(isite);		// IxV1 -> V1
	} else {
	    vp = actuals.restrictWith(r.Z.ithVar(pidx));	// V2
	    vp.replaceWith(V2toV1);				// V1
	}
	BDD vppt = keepV1 ? r.vP.and(vp) : r.vP.relprod(vp, V1set);		// V1cxH1xH1c
	if (r.NNfilter != null) vppt.andWith(r.NNfilter.id());
	// begin work-around
	if (pidx != -1 && formalFilter != null) {
	    vppt.andWith(formalFilter.restrict(r.Z.ithVar(pidx)));
	    formalFilter.free();
	}
	// end work-around
	BDD h0hp = keepV1 ? v0pt.and(vppt) : v0pt.relprod(vppt, V1cset);	// H1xH1cxH2xH2c
	vp.free();
	v0pt.free();
	vppt.free();
	return h0hp;
    }

    /**
     * Implements Vladimir's idea of finding out what types go in a collection.
     *
     * @return BDD H2 x T1 that maps collection objects to the shared supertypes of their elements.
     */
    public TypedBDD findCollectionTypes(boolean trace) {
	PA r = res.r;
	storedin = (TypedBDD)r.bdd.zero(); 		// H1xH1cxH2xH2c
	copiedin = (TypedBDD)r.bdd.zero(); 		// H1xH2
	musthave = (TypedBDD)r.bdd.zero(); 		// H1xH2
	if (!r.CONTEXT_SENSITIVE) {
            System.out.println("Sorry, this analysis has only been debugged in context-sensitive mode");
	    return storedin;
	}
	BDD H1set = r.H1.set();
	BDD H2set = r.H2.set();

	// iterate over all callsites (XXX use a BDD-filter for this instead?)
	for (int iidx = 0; iidx < r.Imap.size(); iidx++) {
	    ProgramLocation call = (ProgramLocation)r.Imap.get(iidx);

	    // is this a call that adds to a collection?
	    Collection entries = getApplicableEntries(call.getTargetMethod());
	    if (entries == null)
		continue;

	    Iterator it = entries.iterator();
	    while (it.hasNext()) {
		CollCall cc = (CollCall)it.next();
		if (trace) {
		    System.out.println("I(" + iidx + ") Z(" + cc.pidx + ") method " 
			+ cc.meth + " " + cc.what + " " + call.toStringLong());
		}

		if (cc.what.equals("consttype") || cc.what.equals("return")) {
		    continue;	// XXX later
		}

		if (cc.pidx.equals("-1")) {
		    continue;
		}

		if (cc.what.equals("addall")) {
		    BDD h0hp = getPointsToForP0andPi(r, call, iidx, cc, false);
		    copiedin.orWith(h0hp);	// H2 has collection, H1 has collection being added
		} else 
		if (cc.what.equals("add")) {
		    BDD h0hp = getPointsToForP0andPi(r, call, iidx, cc, false);
		    if (trace) System.out.println("h0hp is " + res.toString((TypedBDD)h0hp, -1));
		    storedin.orWith(h0hp);	// H2 has collection, H1 has items being added
		} else 
		if (cc.what.equals("return")) {
		} else
		if (cc.what.equals("musthave")) {
		    BDD h0hp = getPointsToForP0andPi(r, call, iidx, cc, true);
		    musthave.orWith(h0hp);	// H2 has collection, H1 has types being passed
		} else {
		    System.out.println("can't handle " + cc.what);
		}
	    }
	}

	TypedBDD tmp = null;

	// propagate the elements of all copied collections to the copy destination
	tmp = (TypedBDD)copiedin.exist(r.H1set);
	BDD old_s = storedin.id();
	for (int cnt = 1;;++cnt) {
	    for (Iterator collections = tmp.iterator(); collections.hasNext(); ) {
		BDD this_col = (BDD)collections.next();			// H2
		BDD other_cols = copiedin.restrict(this_col);		// H1xH2 -> H1
		other_cols.replaceWith(r.H1toH2);			// H1 -> H2

		BDD otherobjs = storedin.relprod(other_cols, H2set);	// H2xH1 x H2 -> H1
		if (otherobjs.isZero())
		    continue;
		otherobjs.andWith(this_col);
		storedin.orWith(otherobjs);
	    }
	    boolean nochange = storedin.equals(old_s);
	    old_s.free();
	    if (nochange)
		break;
	    old_s = storedin.id();
	    if (trace) System.out.println("iteration #" + cnt);
	}
	tmp.free();
	return storedin;
    }

    /** 
     * Determine the supertype of all inserted objects for each collection.
     */
    public TypedBDD determineSupertypes(boolean trace) {
	PA r = res.r;
	BDD H1set = r.H1.set();
	BDD H1cset = r.H1c.set();
	TypedBDD tmp = (TypedBDD)storedin.exist(r.H1set);	// H2xH2c

	// determine the supertype of all inserted items for each collection
	supertypes = (TypedBDD)r.bdd.zero();				// H2 x T1
	for (Iterator collections = tmp.iterator(); collections.hasNext(); ) {
	    BDD c = (BDD)collections.next();			// H2xH2c
	    BDD items = storedin.restrict(c);			// H1xH1cxH2xH2c -> H1xH1c
	    BDD items1 = items.exist(H1cset);
	    items.free();
	    BDD itemtypes = items1.relprod(r.hT, H1set);	// H1 x H1xT2 -> T2
	    items1.free();
	    BDD stypes = res.calculateCommonSupertype(itemtypes);// T2 -> T1
	    itemtypes.free();
	    c.andWith(stypes);					// H2 x T1
	    supertypes.orWith(c);
	}
	if (trace) System.out.println("supertypes after storedin:\n" + res.toString(supertypes, -1));
	tmp.free();
	return (TypedBDD)supertypes;
    }

    public TypedBDD checkMustHaves(boolean trace) {
	PA r = res.r;
	TypedBDD t = (TypedBDD)musthave.apply(storedin.and(r.V1c.domain()).andWith(r.V1.domain()), BDDFactory.diff);	
	return t;
    }

    public TypedBDD checkBadTypes(boolean trace) {
	PA r = res.r;
	/*
	TypedBDD musthave;  // H1xH2		(Objects x Collection)
	TypedBDD supertypes;// H2xT1 		(Collection x Type)
	r.aT: T1xT2 (T2 can be assigned to T1, T2<:T1)
	r.hT: H1xT2, heap type
	*/
	if (trace)
	    System.out.println("musthaves:\n" + res.toString(musthave, -1));
	BDD H2set = r.H2.set();
	BDD H1set = r.H1.set();
	BDD t = musthave.relprod(supertypes, H2set);	// H1xH2xMORE x H2xT1 -> H1xT1xMORE
	BDD t2 = r.hT.relprod(t, H1set);  	// H1xT2 x H1xT1xMORE -> T2xT1xMORE
	t.free();
	BDD pacifyTypedBDD = r.aT.and(r.H1c.domain()).andWith(r.H2c.domain());
	BDD incompatible = t2.apply(pacifyTypedBDD, BDDFactory.diff);
	t2.free();
	return (TypedBDD)incompatible;
    }
}
