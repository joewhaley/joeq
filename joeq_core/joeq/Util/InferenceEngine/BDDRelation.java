// BDDRelation.java, created Mar 16, 2004 12:40:26 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;

/**
 * BDDRelation
 * 
 * @author jwhaley
 * @version $Id$
 */
public class BDDRelation extends Relation {
    
    BDDSolver solver;
    BDD relation;
    List/*<BDDDomain>*/ domains;
    
    /**
     * @param bdd
     */
    public BDDRelation(BDDSolver solver, String name, List fieldNames, List fieldDomains) {
        super(name, fieldNames, fieldDomains);
        this.solver = solver;
        this.relation = solver.bdd.zero();
        this.domains = new LinkedList();
        for (int i = 0; i < fieldDomains.size(); ++i) {
            FieldDomain fd = (FieldDomain) fieldDomains.get(i);
            Collection doms = solver.getBDDDomains(fd);
            BDDDomain d = null;
            for (Iterator j = doms.iterator(); j.hasNext(); ) {
                BDDDomain dom = (BDDDomain) j.next();
                if (!domains.contains(dom)) {
                    d = dom;
                    break;
                }
            }
            if (d == null) {
                d = solver.allocateBDDDomain(fd);
            }
            if (solver.TRACE) solver.out.println("Field "+fieldNames.get(i)+" ("+fieldDomains.get(i)+") assigned to BDDDomain "+d);
            domains.add(d);
        }
    }
    
    public void load() {
        try {
            load(name+".bdd");
            if (solver.NOISY) solver.out.println("Loaded BDD from file: "+name+".bdd");
        } catch (IOException x) {
        }
        
        try {
            loadTuples(name+".tuples");
            if (solver.NOISY) solver.out.println("Loaded tuples from file: "+name+".tuples");
        } catch (IOException x) {
        }
    }
    
    public void load(String filename) throws IOException {
        BDD r2 = solver.bdd.load(filename);
        if (r2 != null) {
            relation.free();
            relation = r2;
        }
    }
    
    public void loadTuples(String filename) throws IOException {
        BufferedReader in = new BufferedReader(new FileReader(filename));
        for (;;) {
            String s = in.readLine();
            if (s == null) break;
            if (s.length() == 0) continue;
            if (s.startsWith("#")) continue;
            StringTokenizer st = new StringTokenizer(s);
            BDD b = solver.bdd.one();
            for (int i = 0; i < domains.size(); ++i) {
                BDDDomain d = (BDDDomain) domains.get(i);
                String v = st.nextToken();
                if (v.equals("*")) {
                    b.andWith(d.domain());
                } else {
                    long l = Long.parseLong(v);
                    b.andWith(d.ithVar(l));
                    if (solver.TRACE) solver.out.print(fieldNames.get(i)+": "+l+", ");
                }
            }
            if (solver.TRACE) solver.out.println();
            relation.orWith(b);
        }
    }
    
    public void save() throws IOException {
        save(name+".rbdd");
    }
    
    public void save(String filename) throws IOException {
        solver.bdd.save(filename, relation);
    }

	public void saveNegated() throws IOException {
		solver.bdd.save("not"+name+".rbdd", relation.not());
	}
    
    public void saveTuples() throws IOException {
        saveTuples(name+".rtuples");
    }
    
    public void saveTuples(String filename) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename));
        BDDDomain iterDomain = (BDDDomain) domains.get(0);
        BDD foo = quantifyOtherDomains(relation, iterDomain);
        for (Iterator i = foo.iterator(iterDomain.set()); i.hasNext(); ) {
            BDD q = (BDD) i.next();
            q.andWith(relation.id());
            while (!q.isZero()) {
                long[] v = q.scanAllVar();
                BDD t = solver.bdd.one();
                for (Iterator j = domains.iterator(); j.hasNext(); ) {
                    BDDDomain d = (BDDDomain) j.next();
                    if (!domains.contains(d)) continue;
                    if (quantifyOtherDomains(q, d).isOne()) {
                        dos.writeBytes("* ");
                        t.andWith(d.domain());
                    } else {
                        dos.writeBytes(v[d.getIndex()]+" ");
                        t.andWith(d.ithVar(v[d.getIndex()]));
                    }
                }
                q.applyWith(t, BDDFactory.diff);
                dos.writeBytes("\n");
            }
        }
        dos.close();
    }
    
    BDD quantifyOtherDomains(BDD q, BDDDomain d) {
        BDD result = q.id();
        for (int i = 0; i < solver.bdd.numberOfDomains(); ++i) {
            if (i == d.getIndex()) continue;
            BDD r2 = result.exist(solver.bdd.getDomain(i).set());
            result.free();
            result = r2;
        }
        return result;
    }
    
}
