// Solver.java, created Mar 16, 2004 7:07:16 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.StringTokenizer;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;

/**
 * Solver
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class Solver {
    
    boolean NOISY = true;
    boolean SPLIT_RULES = true;
    boolean REPORT_STATS = true;
    boolean TRACE = System.getProperty("tracesolve") != null;
    boolean TRACE_FULL = System.getProperty("fulltracesolve") != null;
    PrintStream out = System.out;
    
    abstract InferenceRule createInferenceRule(List/*<RuleTerm>*/ top, RuleTerm bottom);
    abstract Relation createRelation(String name,
                                     List/*<String>*/ names,
                                     List/*<FieldDomain>*/ fieldDomains,
                                     List/*<String>*/ fieldOptions);
    public abstract void initialize();
    public abstract void solve();
    public abstract void finish();
    
    public FieldDomain getFieldDomain(String name) {
        return (FieldDomain) nameToFieldDomain.get(name);
    }
    
    public Relation getRelation(String name) {
        return (Relation) nameToRelation.get(name);
    }
    
    Map/*<String,FieldDomain>*/ nameToFieldDomain;
    Map/*<String,Relation>*/ nameToRelation;
    List/*<InferenceRule>*/ rules;
    Collection/*<Relation>*/ relationsToLoad;
    Collection/*<Relation>*/ relationsToLoadTuples;
    Collection/*<Relation>*/ relationsToDump;
    Collection/*<Relation>*/ relationsToDumpNegated;
    Collection/*<Relation>*/ relationsToDumpTuples;
    
    public static void main(String[] args) throws IOException, InstantiationException, IllegalAccessException, ClassNotFoundException {
        
        String fdFilename = System.getProperty("fielddomains", "fielddomains");
        String relationsFilename = System.getProperty("relations", "relations");
        String rulesFilename = System.getProperty("rules", "rules");
        String solverName = System.getProperty("solver", "joeq.Util.InferenceEngine.BDDSolver");
        
        Solver dis;
        dis = (Solver) Class.forName(solverName).newInstance();
        
        if (dis.NOISY) dis.out.println("Loading field domains from \""+fdFilename+"\"");
        BufferedReader in = new BufferedReader(new FileReader(fdFilename));
        dis.readFieldDomains(in);
        in.close();
        if (dis.NOISY) dis.out.println("Done loading "+dis.nameToFieldDomain.size()+" field domains.");
        
        dis.initialize();
        
        if (dis.NOISY) dis.out.println("Loading relations from \""+relationsFilename+"\"");
        in = new BufferedReader(new FileReader(relationsFilename));
        dis.readRelations(in);
        in.close();
        if (dis.NOISY) dis.out.println("Done loading "+dis.nameToRelation.size()+" relations.");
        
        if (dis.NOISY) dis.out.println("Loading rules from \""+rulesFilename+"\"");
        in = new BufferedReader(new FileReader(rulesFilename));
        dis.readRules(in);
        in.close();
        if (dis.NOISY) dis.out.println("Done loading "+dis.rules.size()+" rules.");
        
        if (dis.NOISY) dis.out.println("Loading initial relations...");
        long time = System.currentTimeMillis();
        dis.loadInitialRelations();
        time = System.currentTimeMillis() - time;
        if (dis.NOISY) dis.out.println("done. ("+time+" ms)");
        
        if (dis.SPLIT_RULES) {
            if (dis.NOISY) dis.out.println("Splitting rules...");
            dis.splitRules();
            if (dis.NOISY) dis.out.println("done.");
        }
        
        if (dis.NOISY) dis.out.println("Solving...");
        time = System.currentTimeMillis();
        dis.solve();
        time = System.currentTimeMillis() - time;
        if (dis.NOISY) dis.out.println("done. ("+time+" ms)");
        
        dis.finish();
        
        if (dis.REPORT_STATS) {
            dis.reportStats();
        }
        
        if (dis.NOISY) dis.out.println("Saving results...");
        time = System.currentTimeMillis();
        dis.saveResults();
        time = System.currentTimeMillis() - time;
        if (dis.NOISY) dis.out.println("done. ("+time+" ms)");
        
    }
    
    void readFieldDomains(BufferedReader in) throws IOException {
        nameToFieldDomain = new HashMap();
        for (;;) {
            String s = in.readLine();
            if (s == null) break;
            if (s.length() == 0) continue;
            if (s.startsWith("#")) continue;
            StringTokenizer st = new StringTokenizer(s);
            FieldDomain fd = readFieldDomain(st);
            if (TRACE) out.println("Loaded field domain "+fd+" size "+fd.size);
            nameToFieldDomain.put(fd.name, fd);
        }
    }
    
    FieldDomain readFieldDomain(StringTokenizer st) throws IOException {
        String name = st.nextToken();
        long size = Long.parseLong(st.nextToken());
        FieldDomain fd = new FieldDomain(name, size);
        if (st.hasMoreTokens()) {
            String mapName = st.nextToken();
            DataInputStream dis = new DataInputStream(new FileInputStream(mapName));
            fd.loadMap(dis);
            dis.close();
        }
        return fd;
    }
    
    void readRelations(BufferedReader in) throws IOException {
        nameToRelation = new HashMap();
        relationsToLoad = new LinkedList();
        relationsToLoadTuples = new LinkedList();
        relationsToDump = new LinkedList();
        relationsToDumpNegated = new LinkedList();
        relationsToDumpTuples = new LinkedList();
        for (;;) {
            String s = in.readLine();
            if (s == null) break;
            if (s.length() == 0) continue;
            if (s.startsWith("#")) continue;
            StringTokenizer st = new StringTokenizer(s);
            Relation r = parseRelation(st);
            if (TRACE) out.println("Loaded relation "+r);
            nameToRelation.put(r.name, r);
        }
    }
    
    Relation parseRelation(StringTokenizer st) {
        String name = st.nextToken();
        String openParen = st.nextToken();
        if (!openParen.equals("(")) throw new IllegalArgumentException();
        List fieldNames = new LinkedList();
        List fieldDomains = new LinkedList();
        List fieldOptions = new LinkedList();
        
        for (;;) {
            String fName = st.nextToken();
            fieldNames.add(fName);
            String colon = st.nextToken();
            if (!colon.equals(":")) throw new IllegalArgumentException("Expected \":\", got \""+colon+"\"");
            String fdName = st.nextToken();
            FieldDomain fd = getFieldDomain(fdName);
            if (fd == null) throw new IllegalArgumentException("Unknown field domain "+fdName);
            fieldDomains.add(fd);
            String comma = st.nextToken();
            if (comma.startsWith("bdd:")) {
                fieldOptions.add(comma.substring(4));
                comma = st.nextToken();
            } else {
                fieldOptions.add("");
            }
            if (comma.equals(")")) break;
            if (!comma.equals(",")) throw new IllegalArgumentException("Expected \",\", got \""+fName+"\"");
        }
        Relation r = createRelation(name, fieldNames, fieldDomains, fieldOptions);
        while (st.hasMoreTokens()) {
            String option = st.nextToken();
            if (option.equals("save")) {
                relationsToDump.add(r);
            } else if (option.equals("savenot")) {
                relationsToDumpNegated.add(r);
            } else if (option.equals("savetuples")) {
                relationsToDumpTuples.add(r);
            } else if (option.equals("load")) {
                relationsToLoad.add(r);
            } else if (option.equals("loadtuples")) {
                relationsToLoadTuples.add(r);
            } else {
                throw new IllegalArgumentException("Unexpected option '"+option+"'");
            }
        }
        return r;
    }
    
    void readRules(BufferedReader in) throws IOException {
        rules = new LinkedList();
        for (;;) {
            String s = in.readLine();
            if (s == null) break;
            if (s.length() == 0) continue;
            if (s.startsWith("#")) continue;
            StringTokenizer st = new StringTokenizer(s);
            InferenceRule r = parseRule(st);
            if (TRACE) out.println("Loaded rule(s) "+r);
            rules.add(r);
        }
    }
    
    InferenceRule parseRule(StringTokenizer st) {
        Map/*<String,Variable>*/ nameToVar = new HashMap();
        List/*<RuleTerm>*/ terms = new LinkedList();
        for (;;) {
            RuleTerm rt = parseRuleTerm(nameToVar, st);
            terms.add(rt);
            String sep = st.nextToken();
            if (sep.equals("/")) break;
            if (!sep.equals(",")) throw new IllegalArgumentException();
        }
        RuleTerm bottom = parseRuleTerm(nameToVar, st);
        InferenceRule ir = createInferenceRule(terms, bottom);
        return ir;
    }
    
    RuleTerm parseRuleTerm(Map/*<String,Variable>*/ nameToVar, StringTokenizer st) {
        String openParen = st.nextToken();
        if (!openParen.equals("(")) throw new IllegalArgumentException("Expected '(', got '"+openParen+"'");
        List/*<Object>*/ vars = new LinkedList();
        for (;;) {
            String varName = st.nextToken();
            char firstChar = varName.charAt(0);
            Object var;
            if (firstChar >= '0' && firstChar <= '9') {
                var = new Constant(Long.parseLong(varName));
            } else if (firstChar == '"') {
                String namedConstant = varName.substring(1, varName.length()-1);
                var = namedConstant;
            } else if (!varName.equals("_")) {
                var = (Variable) nameToVar.get(varName);
                if (var == null) nameToVar.put(varName, var = new Variable(varName));
            } else {
                var = new Variable();
            }
            if (vars.contains(var)) throw new IllegalArgumentException("Duplicate variable "+var);
            vars.add(var);
            String sep = st.nextToken();
            if (sep.equals(")")) break;
            if (!sep.equals(",")) throw new IllegalArgumentException("Expected ',' or ')', got '"+sep+"'");
        }
        String in = st.nextToken();
        if (!in.equals("in")) throw new IllegalArgumentException();
        String relationName = st.nextToken();
        Relation r = getRelation(relationName);
        if (r == null) throw new IllegalArgumentException("Unknown relation "+relationName);
        if (r.fieldDomains.size() != vars.size()) throw new IllegalArgumentException();
        
        int n = 0;
        for (ListIterator li = vars.listIterator(); li.hasNext(); ++n) {
            Object var = li.next();
            if (var instanceof String) {
                String namedConstant = (String) var;
                FieldDomain fd = (FieldDomain) r.fieldDomains.get(n);
                Variable constant = new Constant(fd.namedConstant(namedConstant));
                li.set(constant);
            }
        }
        for (int i = 0; i < r.fieldDomains.size(); ++i) {
            Variable var = (Variable) vars.get(i);
            FieldDomain fd = (FieldDomain) r.fieldDomains.get(i);
            if (var.fieldDomain == null) var.fieldDomain = fd;
            else if (var.fieldDomain != fd) throw new IllegalArgumentException();
        }
        RuleTerm rt = new RuleTerm(vars, r);
        return rt;
    }
    
//    public Relation getOrCreateRelation(String name, List/*<Variable>*/ vars) {
//        Relation r = (Relation) nameToRelation.get(name);
//        if (r == null) nameToRelation.put(name, r = createRelation(name, vars));
//        return r;
//    }
    
    void loadInitialRelations() throws IOException {
        for (Iterator i = relationsToLoad.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            r.load();
        }
        for (Iterator i = relationsToLoadTuples.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            r.loadTuples();
        }
    }
    
    void splitRules() {
        List newRules = new LinkedList();
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule r = (InferenceRule) i.next();
            newRules.addAll(r.split(this));
        }
        rules.addAll(newRules);
    }
    
    void reportStats() {
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule r = (InferenceRule) i.next();
            r.reportStats();
        }
    }
    
    void saveResults() throws IOException {
        for (Iterator i = relationsToDump.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            if (NOISY) out.println("Dumping BDD for "+r);
            r.save();
        }
        for (Iterator i = relationsToDumpNegated.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            if (NOISY) out.println("Dumping negated BDD for "+r);
            r.saveNegated();
        }
        for (Iterator i = relationsToDumpTuples.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            if (NOISY) out.println("Dumping tuples for "+r);
            r.saveTuples();
        }
    }

}
