// Solver.java, created Mar 16, 2004 7:07:16 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;

import joeq.Util.Collections.Pair;

/**
 * Solver
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class Solver {
    
    boolean NOISY = !System.getProperty("noisy", "yes").equals("no");
    boolean SPLIT_ALL_RULES = false;
    boolean REPORT_STATS = true;
    boolean TRACE = System.getProperty("tracesolve") != null;
    boolean TRACE_FULL = System.getProperty("fulltracesolve") != null;
    PrintStream out = System.out;
    
    abstract InferenceRule createInferenceRule(List/*<RuleTerm>*/ top, RuleTerm bottom);
    abstract Relation createEquivalenceRelation(FieldDomain fd);
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
    Map/*<FieldDomain,Relation>*/ equivalenceRelations;
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
        
        if (dis.NOISY) dis.out.println("Splitting rules...");
        dis.splitRules();
        if (dis.NOISY) dis.out.println("done.");
        
        dis.out.println("Solving...");
        time = System.currentTimeMillis();
        dis.solve();
        time = System.currentTimeMillis() - time;
        dis.out.println("done. ("+time+" ms)");
        
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
        String name = nextToken(st);
        long size = Long.parseLong(nextToken(st));
        FieldDomain fd = new FieldDomain(name, size);
        if (st.hasMoreTokens()) {
            String mapName = nextToken(st);
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
            StringTokenizer st = new StringTokenizer(s, " (:,)", true);
            Relation r = parseRelation(st);
            if (TRACE) out.println("Loaded relation "+r);
            nameToRelation.put(r.name, r);
        }
    }
    
    static String nextToken(StringTokenizer st) {
        String s;
        do {
            s = st.nextToken();
        } while (s.equals(" "));
        return s;
    }
    
    Relation parseRelation(StringTokenizer st) {
        String name = nextToken(st);
        String openParen = nextToken(st);
        List fieldNames = new LinkedList();
        List fieldDomains = new LinkedList();
        List fieldOptions = new LinkedList();
        
        for (;;) {
            String fName = nextToken(st);
            fieldNames.add(fName);
            String colon = nextToken(st);
            if (!colon.equals(":")) throw new IllegalArgumentException("Expected \":\", got \""+colon+"\"");
            String fdName = nextToken(st);
            FieldDomain fd = getFieldDomain(fdName);
            if (fd == null) throw new IllegalArgumentException("Unknown field domain "+fdName);
            fieldDomains.add(fd);
            String comma = nextToken(st);
            if (comma.startsWith("bdd=")) {
                fieldOptions.add(comma.substring(4));
                comma = nextToken(st);
            } else {
                fieldOptions.add("");
            }
            if (comma.equals(")")) break;
            if (!comma.equals(",")) throw new IllegalArgumentException("Expected \",\", got \""+fName+"\"");
        }
        Relation r = createRelation(name, fieldNames, fieldDomains, fieldOptions);
        while (st.hasMoreTokens()) {
            String option = nextToken(st);
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
            StringTokenizer st = new StringTokenizer(s, " (,/).", true);
            InferenceRule r = parseRule(st);
            if (TRACE) out.println("Loaded rule(s) "+r);
            else out.print('.');
            rules.add(r);
        }
        out.println();
    }
    
    InferenceRule parseRule(StringTokenizer st) {
        Map/*<String,Variable>*/ nameToVar = new HashMap();
        RuleTerm bottom = parseRuleTerm(nameToVar, st);
        String sep = nextToken(st);
        if (!sep.equals(":-"))
            throw new IllegalArgumentException("Expected \":-\", got \""+sep+"\"");
        List/*<RuleTerm>*/ terms = new LinkedList();
        for (;;) {
            RuleTerm rt = parseRuleTerm(nameToVar, st);
            if (rt == null) break;
            terms.add(rt);
            sep = nextToken(st);
            if (sep.equals(".")) break;
            if (!sep.equals(",")) throw new IllegalArgumentException();
        }
        InferenceRule ir = createInferenceRule(terms, bottom);
        parseRuleOptions(ir, st);
        return ir;
    }
    
    void parseRuleOptions(InferenceRule ir, StringTokenizer st) {
        while (st.hasMoreTokens()) {
            String option = nextToken(st);
            if (option.equals("split")) {
                if (NOISY) out.println("Splitting rule "+ir);
                ir.split = true;
            } else if (option.equals("cacheafterrename")) {
                BDDInferenceRule r = (BDDInferenceRule) ir;
                r.cache_before_rename = false;
            } else if (option.equals("testorder")) {
                BDDInferenceRule r = (BDDInferenceRule) ir;
                r.test_order = true;
            } else {
                throw new IllegalArgumentException("Unknown rule option \""+option+"\"");
            }
        }
    }
    
    RuleTerm parseRuleTerm(Map/*<String,Variable>*/ nameToVar, StringTokenizer st) {
        String relationName = nextToken(st);
        String openParen = nextToken(st);
        if (openParen.equals("=")) {
            // "a = b".
            FieldDomain fd = null;
            Variable var1 = (Variable) nameToVar.get(relationName);
            if (var1 == null) nameToVar.put(relationName, var1 = new Variable(relationName));
            else fd = var1.fieldDomain;
            String varName2 = nextToken(st);
            Variable var2 = (Variable) nameToVar.get(varName2);
            if (var2 == null) nameToVar.put(varName2, var2 = new Variable(varName2));
            else {
                FieldDomain fd2 = var2.fieldDomain;
                if (fd == null) fd = fd2;
                else if (fd != fd2)
                    throw new IllegalArgumentException("Variable "+var1+" and "+var2+" have different field domains.");
            }
            if (fd == null)
                throw new IllegalArgumentException("Cannot use \"=\" on two unbound variables.");
            Relation r = getEquivalenceRelation(fd);
            List vars = new Pair(var1, var2);
            RuleTerm rt = new RuleTerm(vars, r);
            return rt;
        } else if (!openParen.equals("("))
            throw new IllegalArgumentException("Expected \"(\" or \"=\", got \""+openParen+"\"");

        Relation r = getRelation(relationName);
        if (r == null)
            throw new IllegalArgumentException("Unknown relation "+relationName);
        List/*<Variable>*/ vars = new LinkedList();
        for (;;) {
            FieldDomain fd = (FieldDomain) r.fieldDomains.get(vars.size()); 
            String varName = nextToken(st);
            char firstChar = varName.charAt(0);
            Variable var;
            if (firstChar >= '0' && firstChar <= '9') {
                var = new Constant(Long.parseLong(varName));
            } else if (firstChar == '"') {
                String namedConstant = varName.substring(1, varName.length()-1);
                var = new Constant(fd.namedConstant(namedConstant));
            } else if (!varName.equals("_")) {
                var = (Variable) nameToVar.get(varName);
                if (var == null) nameToVar.put(varName, var = new Variable(varName));
            } else {
                var = new Variable();
            }
            if (vars.contains(var)) throw new IllegalArgumentException("Duplicate variable "+var);
            vars.add(var);
            if (var.fieldDomain == null) var.fieldDomain = fd;
            else if (var.fieldDomain != fd) throw new IllegalArgumentException("Variable "+var+" used as both "+var.fieldDomain+" and "+fd);
            String sep = nextToken(st);
            if (sep.equals(")")) break;
            if (!sep.equals(",")) throw new IllegalArgumentException("Expected ',' or ')', got '"+sep+"'");
        }
        if (r.fieldDomains.size() != vars.size())
            throw new IllegalArgumentException("Wrong number of vars in rule term for "+relationName);
        
        RuleTerm rt = new RuleTerm(vars, r);
        return rt;
    }
    
//    public Relation getOrCreateRelation(String name, List/*<Variable>*/ vars) {
//        Relation r = (Relation) nameToRelation.get(name);
//        if (r == null) nameToRelation.put(name, r = createRelation(name, vars));
//        return r;
//    }
    
    Relation getEquivalenceRelation(FieldDomain fd) {
        Relation r = (Relation) equivalenceRelations.get(fd);
        if (r == null) {
            equivalenceRelations.put(fd, createEquivalenceRelation(fd));
        }
        return r;
    }
    
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
            if (SPLIT_ALL_RULES || r.split)
                newRules.addAll(r.split(rules.indexOf(r), this));
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
