// Solver.java, created Mar 16, 2004 7:07:16 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.InferenceEngine;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * Solver
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class Solver {
    
    boolean TRACE = true;
    PrintStream out = System.out;
    
    abstract InferenceRule createInferenceRule(List/*<RuleTerm>*/ top, RuleTerm bottom);
    abstract Relation createRelation(String name, List/*<String>*/ names, List/*<FieldDomain>*/ fieldDomains);
    public abstract void solve();
    
    public FieldDomain getFieldDomain(String name) {
        return (FieldDomain) nameToFieldDomain.get(name);
    }
    
    public Relation getRelation(String name) {
        return (Relation) nameToRelation.get(name);
    }
    
    Map/*<String,FieldDomain>*/ nameToFieldDomain;
    Map/*<String,Relation>*/ nameToRelation;
    List/*<InferenceRule>*/ rules;
    
    public static void main(String[] args) throws IOException, InstantiationException, IllegalAccessException, ClassNotFoundException {
        
        String fdFilename = System.getProperty("fielddomains", "fielddomains");
        String relationsFilename = System.getProperty("relations", "relations");
        String rulesFilename = System.getProperty("rules", "rules");
        String solverName = System.getProperty("solver", "joeq.Util.InferenceEngine.BDDSolver");
        
        Solver dis;
        dis = (Solver) Class.forName(solverName).newInstance();
        
        if (dis.TRACE) dis.out.println("Loading field domains from \""+fdFilename+"\"");
        BufferedReader in = new BufferedReader(new FileReader(fdFilename));
        dis.readFieldDomains(in);
        in.close();
        if (dis.TRACE) dis.out.println("Done loading "+dis.nameToFieldDomain.size()+" field domains.");
        
        if (dis.TRACE) dis.out.println("Loading relations from \""+relationsFilename+"\"");
        in = new BufferedReader(new FileReader(relationsFilename));
        dis.readRelations(in);
        in.close();
        if (dis.TRACE) dis.out.println("Done loading "+dis.nameToRelation.size()+" relations.");
        
        if (dis.TRACE) dis.out.println("Loading rules from \""+rulesFilename+"\"");
        in = new BufferedReader(new FileReader(rulesFilename));
        dis.readRules(in);
        in.close();
        if (dis.TRACE) dis.out.println("Done loading "+dis.rules.size()+" rules.");
        
        dis.loadInitialRelations();
        
        dis.solve();
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
        for (;;) {
            String fName = st.nextToken();
            fieldNames.add(fName);
            String colon = st.nextToken();
            if (!colon.equals(":")) throw new IllegalArgumentException();
            String fdName = st.nextToken();
            FieldDomain fd = getFieldDomain(fdName);
            if (fd == null) throw new IllegalArgumentException("Unknown field domain "+fdName);
            fieldDomains.add(fd);
            String comma = st.nextToken();
            if (comma.equals(")")) break;
            if (!comma.equals(",")) throw new IllegalArgumentException("Expected \",\", got \""+fName+"\"");
        }
        Relation r = createRelation(name, fieldNames, fieldDomains);
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
            if (TRACE) out.println("Loaded rule "+r);
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
        if (!openParen.equals("(")) throw new IllegalArgumentException();
        List/*<Variable>*/ vars = new LinkedList();
        for (;;) {
            String varName = st.nextToken();
            Variable var;
            if (!varName.equals("_")) {
                var = (Variable) nameToVar.get(varName);
                if (var == null) nameToVar.put(varName, var = new Variable(varName));
            } else {
                var = new Variable();
            }
            if (vars.contains(var)) throw new IllegalArgumentException("Duplicate variable "+var);
            vars.add(var);
            String sep = st.nextToken();
            if (sep.equals(")")) break;
            if (!sep.equals(",")) throw new IllegalArgumentException();
        }
        String in = st.nextToken();
        if (!in.equals("in")) throw new IllegalArgumentException();
        String relationName = st.nextToken();
        Relation r = getRelation(relationName);
        if (r == null) throw new IllegalArgumentException("Unknown relation "+relationName);
        if (r.fieldDomains.size() != vars.size()) throw new IllegalArgumentException();
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
    
    void loadInitialRelations() {
        for (Iterator i = nameToRelation.values().iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            r.load();
        }
    }
}
