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
    
    public BDDRelation(BDDSolver solver, String name, List fieldNames, List fieldDomains, List fieldOptions) {
        super(name, fieldNames, fieldDomains);
        this.solver = solver;
        this.relation = solver.bdd.zero();
        this.domains = new LinkedList();
        System.out.println("Constructing BDDRelation "+name+" with "+fieldDomains.size()+" domains "+fieldNames.size()+" names.");
        for (int i = 0; i < fieldDomains.size(); ++i) {
            FieldDomain fd = (FieldDomain) fieldDomains.get(i);
            Collection doms = solver.getBDDDomains(fd);
            BDDDomain d = null;
            String option = (String) fieldOptions.get(i);
            if (option.length() > 0) {
                // use the given domain.
                if (!option.startsWith(fd.name))
                    throw new IllegalArgumentException("Field "+name+" has domain "+fd+", but tried to assign "+option);
                int index = Integer.parseInt(option.substring(fd.name.length()));
                for (Iterator j = doms.iterator(); j.hasNext(); ) {
                    BDDDomain dom = (BDDDomain) j.next();
                    if (dom.getName().equals(option)) {
                        if (domains.contains(dom))
                            throw new IllegalArgumentException("Cannot assign "+dom+" to field "+name+": "+dom+" is already assigned");
                        d = dom;
                        break;
                    }
                }
                while (d == null) {
                    BDDDomain dom = solver.allocateBDDDomain(fd);
                    if (dom.getName().equals(option)) {
                        d = dom;
                        break;
                    }
                }
            } else {
                // find an applicable domain.
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
            }
            if (solver.TRACE) solver.out.println("Field "+fieldNames.get(i)+" ("+fieldDomains.get(i)+") assigned to BDDDomain "+d);
            domains.add(d);
        }
    }
    
    public void load() {
        try {
            load(name+".bdd");
            if (solver.NOISY) solver.out.println("Loaded BDD from file: "+name+".bdd");
            if (solver.TRACE) solver.out.println("Domains of loaded relation:"+activeDomains());
        } catch (IOException x) {
        }
        
        try {
            loadTuples(name+".tuples");
            if (solver.NOISY) solver.out.println("Loaded tuples from file: "+name+".tuples");
            if (solver.TRACE) solver.out.println("Domains of loaded relation:"+activeDomains());
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
                    if (solver.TRACE_FULL) solver.out.print(fieldNames.get(i)+": "+l+", ");
                }
            }
            if (solver.TRACE_FULL) solver.out.println();
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
    
    public void saveTuples(String fileName) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(fileName));
        int[] a = relation.support().scanSetDomains();
        BDD allDomains = solver.bdd.one();
        System.out.print(fileName+" domains {");
        for (int i = 0; i < a.length; ++i) {
            BDDDomain d = solver.bdd.getDomain(i);
            System.out.print(" "+d.toString());
            allDomains.andWith(d.set());
        }
        System.out.println(" ) = "+relation.nodeCount()+" nodes");
        BDDDomain iterDomain = (BDDDomain) domains.get(0);
        BDD foo = quantifyOtherDomains(relation, iterDomain);
        int lines = 0;
        for (Iterator i = foo.iterator(iterDomain.set()); i.hasNext(); ) {
            BDD q = (BDD) i.next();
            q.andWith(relation.id());
            while (!q.isZero()) {
                BDD sat = q.satOne(allDomains, solver.bdd.zero());
                BDD sup = q.support();
                int[] b = sup.scanSetDomains();
                sup.free();
                long[] v = sat.scanAllVar();
                sat.free();
                BDD t = solver.bdd.one();
                for (int j = 0, k = 0; j < solver.bdd.numberOfDomains(); ++j) {
                    BDDDomain d = solver.bdd.getDomain(j);
                    if (k >= a.length || a[k] != j) {
                        dos.writeBytes("* ");
                        t.andWith(d.domain());
                        continue;
                    } else {
                        ++k;
                    }
                    if (v[j] == 0) {
                        BDD qs = q.support();
                        qs.orWith(d.set());
                        boolean contains = qs.isOne();
                        qs.free();
                        if (!contains) {
                            dos.writeBytes("* ");
                            t.andWith(d.domain());
                            continue;
                        }
                    }
                    dos.writeBytes(v[j]+" ");
                    t.andWith(d.ithVar(v[j]));
                }
                q.applyWith(t, BDDFactory.diff);
                dos.writeBytes("\n");
                ++lines;
            }
            q.free();
        }
        dos.close();
        System.out.println("Done writing "+lines+" lines.");
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
    
    public String activeDomains() {
        BDD s = relation.support();
        int[] a = s.scanSetDomains();
        s.free();
        if (a == null) return "(none)";
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < a.length; ++i) {
            sb.append(solver.bdd.getDomain(a[i]));
            if (i < a.length-1) sb.append(',');
        }
        return sb.toString();
    }
}
