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

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintStream;
import java.math.BigInteger;
import java.text.DecimalFormat;

import joeq.Util.Collections.Pair;
import joeq.Util.IO.MyStringTokenizer;
import joeq.Util.IO.SystemProperties;

/**
 * Solver
 * 
 * @author jwhaley
 * @version $Id$
 */
public abstract class Solver {
    static { SystemProperties.read("solver.properties"); }
    
    boolean NOISY = !System.getProperty("noisy", "yes").equals("no");
    boolean SPLIT_ALL_RULES = false;
    boolean REPORT_STATS = true;
    boolean TRACE = System.getProperty("tracesolve") != null;
    boolean TRACE_FULL = System.getProperty("fulltracesolve") != null;
    PrintStream out = System.out;
    
    abstract InferenceRule createInferenceRule(List/*<RuleTerm>*/ top, RuleTerm bottom);
    abstract Relation createEquivalenceRelation(FieldDomain fd);
    abstract Relation createNotEquivalenceRelation(FieldDomain fd);
    abstract Relation createRelation(String name,
                                     List/*<String>*/ names,
                                     List/*<FieldDomain>*/ fieldDomains,
                                     List/*<String>*/ fieldOptions);
    
    Solver() {
        clear();
    }
    
    public void clear() {
        nameToFieldDomain = new HashMap();
        nameToRelation = new HashMap();
        equivalenceRelations = new HashMap();
        notequivalenceRelations = new HashMap();
        rules = new LinkedList();
        relationsToLoad = new LinkedList();
        relationsToLoadTuples = new LinkedList();
        relationsToDump = new LinkedList();
        relationsToDumpNegated = new LinkedList();
        relationsToDumpTuples = new LinkedList();
        relationsToDumpNegatedTuples = new LinkedList();
        relationsToPrintSize = new LinkedList();
    }
    
    public void initialize() {
        for (Iterator i = nameToRelation.values().iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            r.initialize();
        }
        for (Iterator i = equivalenceRelations.values().iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            r.initialize();
        }
        for (Iterator i = notequivalenceRelations.values().iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            r.initialize();
        }
        for (Iterator i = rules.iterator(); i.hasNext(); ) {
            InferenceRule r = (InferenceRule) i.next();
            r.initialize();
        }
    }
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
    Map/*<FieldDomain,Relation>*/ notequivalenceRelations;
    List/*<InferenceRule>*/ rules;
    Collection/*<Relation>*/ relationsToLoad;
    Collection/*<Relation>*/ relationsToLoadTuples;
    Collection/*<Relation>*/ relationsToDump;
    Collection/*<Relation>*/ relationsToDumpNegated;
    Collection/*<Relation>*/ relationsToDumpTuples;
    Collection/*<Relation>*/ relationsToDumpNegatedTuples;
    Collection/*<Relation>*/ relationsToPrintSize;
    
    public static void main(String[] args) throws IOException, InstantiationException, IllegalAccessException, ClassNotFoundException {
        
        String inputFilename = System.getProperty("datalog");
        if (args.length > 0) inputFilename = args[0];
        
        String solverName = System.getProperty("solver", "joeq.Util.InferenceEngine.BDDSolver");
        
        Solver dis;
        dis = (Solver) Class.forName(solverName).newInstance();
        
        if (dis.NOISY) dis.out.println("Opening Datalog program \""+inputFilename+"\"");
        MyReader in = new MyReader(new LineNumberReader(new FileReader(inputFilename)));
        dis.readDatalogProgram(in);
        
        if (dis.NOISY) dis.out.println(dis.nameToFieldDomain.size()+" field domains.");
        if (dis.NOISY) dis.out.println(dis.nameToRelation.size()+" relations.");
        if (dis.NOISY) dis.out.println(dis.rules.size()+" rules.");
        
        in.close();
        
        if (dis.NOISY) dis.out.print("Splitting rules: ");
        dis.splitRules();
        if (dis.NOISY) dis.out.println("done.");
        
        if (dis.NOISY) dis.out.print("Initializing solver: ");
        dis.initialize();
        if (dis.NOISY) dis.out.println("done.");
        
        if (dis.NOISY) dis.out.print("Loading initial relations: ");
        long time = System.currentTimeMillis();
        dis.loadInitialRelations();
        time = System.currentTimeMillis() - time;
        if (dis.NOISY) dis.out.println("done. ("+time+" ms)");
        
        dis.out.println("Solving: ");
        time = System.currentTimeMillis();
        dis.solve();
        time = System.currentTimeMillis() - time;
        dis.out.println("done. ("+time+" ms)");
        long solveTime = time;
        
        dis.finish();
        
        if (dis.REPORT_STATS) {
            System.out.println("SOLVE_TIME="+solveTime);
            dis.reportStats();
        }
        
        if (dis.NOISY) dis.out.print("Saving results: ");
        time = System.currentTimeMillis();
        dis.saveResults();
        time = System.currentTimeMillis() - time;
        if (dis.NOISY) dis.out.println("done. ("+time+" ms)");
        
    }
    
    public static class MyReader {
        List readerStack = new LinkedList();
        LineNumberReader current;
        
        public MyReader(LineNumberReader r) {
            current = r;
        }
        
        public void registerReader(LineNumberReader r) {
            if (current != null) readerStack.add(current);
            current = r;
        }
        
        public String readLine() throws IOException {
            String s;
            for (;;) {
                s = current.readLine();
                if (s != null) return s;
                if (readerStack.isEmpty()) return null;
                current = (LineNumberReader) readerStack.remove(readerStack.size()-1);
            }
        }
        
        public int getLineNumber() {
            return current.getLineNumber();
        }
        
        public void close() throws IOException {
            for (;;) {
                current.close();
                if (readerStack.isEmpty()) return;
                current = (LineNumberReader) readerStack.remove(readerStack.size()-1);
            }
        }
    }
    
    static String nextToken(MyStringTokenizer st) {
        String s;
        do {
            s = st.nextToken();
        } while (s.equals(" "));
        return s;
    }
    
    static String readLine(MyReader in) throws IOException {
        String s = in.readLine();
        if (s == null) return null;
        s = s.trim();
        while (s.endsWith("\\")) {
            String s2 = in.readLine();
            if (s2 == null) break;
            s2 = s2.trim();
            s += s2;
        }
        return s;
    }
    
    void readDatalogProgram(MyReader in) throws IOException {
        for (;;) {
            String s = readLine(in);
            if (s == null) break;
            if (s.length() == 0) continue;
            if (s.startsWith("#")) continue;
            int lineNum = in.getLineNumber();
            if (s.startsWith(".")) {
                // directive
                parseDirective(in, lineNum, s);
                continue;
            }
            MyStringTokenizer st = new MyStringTokenizer(s);
            if (st.hasMoreTokens()) {
                st.nextToken(); // name
                if (st.hasMoreTokens()) {
                    String num = st.nextToken();
                    boolean isNumber;
                    try {
                        new BigInteger(num);
                        isNumber = true;
                    } catch (NumberFormatException x) {
                        isNumber = false;
                    }
                    if (isNumber) {
                        // field domain
                        FieldDomain fd = parseFieldDomain(lineNum, s);
                        if (TRACE) out.println("Parsed field domain "+fd+" size "+fd.size);
                        nameToFieldDomain.put(fd.name, fd);
                        continue;
                    }
                }
            }
            
            if (s.indexOf(".") > 0) {
                // rule
                InferenceRule ir = parseRule(lineNum, s);
                if (TRACE) out.println("Parsed rule "+ir);
                rules.add(ir);
                continue;
            } else {
                // relation
                Relation r = parseRelation(lineNum, s);
                if (TRACE) out.println("Parsed relation "+r);
                nameToRelation.put(r.name, r);
                continue;
            }
        }
    }
    
    void outputError(int linenum, int colnum, String line, String msg) {
        System.err.println("Error on line "+linenum+":");
        System.err.println(line);
        while (--colnum >= 0) System.err.print(' ');
        System.err.println("^");
        System.err.println(msg);
    }
    
    void parseDirective(MyReader in, int lineNum, String s) throws IOException {
        if (s.startsWith(".include")) {
            int index = ".include".length()+1;
            String fileName = s.substring(index).trim();
            if (fileName.startsWith("\"")) {
                if (!fileName.endsWith("\"")) {
                    outputError(lineNum, index, s, "Unmatched quotes");
                    throw new IllegalArgumentException();
                }
                fileName = fileName.substring(1, fileName.length()-1);
            }
            in.registerReader(new LineNumberReader(new FileReader(fileName)));
        } else {
            outputError(lineNum, 0, s, "Unknown directive \""+s+"\"");
            throw new IllegalArgumentException();
        }
    }
    
    FieldDomain parseFieldDomain(int lineNum, String s) throws IOException {
        MyStringTokenizer st = new MyStringTokenizer(s);
        String name = nextToken(st);
        String num = nextToken(st);
        long size;
        try {
            size = Long.parseLong(num);
        } catch (NumberFormatException x) {
            outputError(lineNum, st.getPosition(), s, "Expected a number, got \""+num+"\"");
            throw new IllegalArgumentException();
        }
        FieldDomain fd = new FieldDomain(name, size);
        if (st.hasMoreTokens()) {
            String mapName = nextToken(st);
            DataInputStream dis = new DataInputStream(new FileInputStream(mapName));
            fd.loadMap(dis);
            dis.close();
        }
        return fd;
    }
    
    Relation parseRelation(int lineNum, String s) {
        MyStringTokenizer st = new MyStringTokenizer(s, " (:,)", true);
        String name = nextToken(st);
        String openParen = nextToken(st);
        if (!openParen.equals("(")) {
            outputError(lineNum, st.getPosition(), s, "Expected \"(\", got \""+openParen+"\"");
            throw new IllegalArgumentException();
        }
        List fieldNames = new LinkedList();
        List fieldDomains = new LinkedList();
        List fieldOptions = new LinkedList();
        
        for (;;) {
            String fName = nextToken(st);
            fieldNames.add(fName);
            String colon = nextToken(st);
            if (!colon.equals(":")) {
                outputError(lineNum, st.getPosition(), s, "Expected \":\", got \""+colon+"\"");
                throw new IllegalArgumentException();
            }
            String fdName = nextToken(st);
            int numIndex = fdName.length() - 1;
            for (;;) {
                char c = fdName.charAt(numIndex);
                if (c < '0' || c > '9') break;
                --numIndex;
                if (numIndex < 0) {
                    outputError(lineNum, st.getPosition(), s, "Expected field domain name, got \""+fdName+"\"");
                    throw new IllegalArgumentException();
                }
            }
            ++numIndex;
            int fdNum = -1;
            if (numIndex < fdName.length()) {
                String number = fdName.substring(numIndex);
                try {
                    fdNum = Integer.parseInt(number);
                } catch (NumberFormatException x) {
                    outputError(lineNum, st.getPosition(), s, "Cannot parse field domain number \""+number+"\"");
                    throw new IllegalArgumentException();
                }
                fdName = fdName.substring(0, numIndex);
            }
            FieldDomain fd = getFieldDomain(fdName);
            if (fd == null) {
                outputError(lineNum, st.getPosition(), s, "Unknown field domain "+fdName);
                throw new IllegalArgumentException();
            }
            fieldDomains.add(fd);
            if (fdNum != -1)
                fieldOptions.add(fdName+fdNum);
            else
                fieldOptions.add("");
            String comma = nextToken(st);
            if (comma.equals(")")) break;
            if (!comma.equals(",")) {
                outputError(lineNum, st.getPosition(), s, "Expected \",\" or \")\", got \""+comma+"\"");
                throw new IllegalArgumentException();
            }
        }
        Relation r = createRelation(name, fieldNames, fieldDomains, fieldOptions);
        while (st.hasMoreTokens()) {
            String option = nextToken(st);
            if (option.equals("output")) {
                relationsToDump.add(r);
            } else if (option.equals("outputnot")) {
                relationsToDumpNegated.add(r);
            } else if (option.equals("outputtuples")) {
                relationsToDumpTuples.add(r);
            } else if (option.equals("outputnottuples")) {
                relationsToDumpNegatedTuples.add(r);
            } else if (option.equals("input")) {
                relationsToLoad.add(r);
            } else if (option.equals("inputtuples")) {
                relationsToLoadTuples.add(r);
            } else if (option.equals("printsize")) {
                relationsToPrintSize.add(r);
            } else {
                outputError(lineNum, st.getPosition(), s, "Unexpected option '"+option+"'");
                throw new IllegalArgumentException();
            }
        }
        return r;
    }
    
    InferenceRule parseRule(int lineNum, String s) {
        MyStringTokenizer st = new MyStringTokenizer(s, " (,/).=!", true);
        Map/*<String,Variable>*/ nameToVar = new HashMap();
        RuleTerm bottom = parseRuleTerm(lineNum, s, nameToVar, st);
        String sep = nextToken(st);
        List/*<RuleTerm>*/ terms = new LinkedList();
        if (!sep.equals(".")) {
            if (!sep.equals(":-")) {
                outputError(lineNum, st.getPosition(), s, "Expected \":-\", got \""+sep+"\"");
                throw new IllegalArgumentException();
            }
            for (;;) {
                RuleTerm rt = parseRuleTerm(lineNum, s, nameToVar, st);
                if (rt == null) break;
                terms.add(rt);
                sep = nextToken(st);
                if (sep.equals(".")) break;
                if (!sep.equals(",")) {
                    outputError(lineNum, st.getPosition(), s, "Expected \".\" or \",\", got \""+sep+"\"");
                    throw new IllegalArgumentException();
                }
            }
        }
        InferenceRule ir = createInferenceRule(terms, bottom);
        parseRuleOptions(lineNum, s, ir, st);
        return ir;
    }
    
    void parseRuleOptions(int lineNum, String s, InferenceRule ir, MyStringTokenizer st) {
        while (st.hasMoreTokens()) {
            String option = nextToken(st);
            if (option.equals("split")) {
                if (TRACE) out.println("Splitting rule "+ir);
                ir.split = true;
            } else if (option.equals("cacheafterrename")) {
                BDDInferenceRule r = (BDDInferenceRule) ir;
                r.cache_before_rename = false;
            } else if (option.equals("findbestorder")) {
                BDDInferenceRule r = (BDDInferenceRule) ir;
                r.find_best_order = true;
            } else {
                // todo: pri=#, maxiter=#
                outputError(lineNum, st.getPosition(), s, "Unknown rule option \""+option+"\"");
                throw new IllegalArgumentException();
            }
        }
    }
    
    RuleTerm parseRuleTerm(int lineNum, String s, Map/*<String,Variable>*/ nameToVar, MyStringTokenizer st) {
        String relationName = nextToken(st);
        String openParen = nextToken(st);
        boolean flip = false;
        if (openParen.equals("!")) {
            flip = true; openParen = nextToken(st);
        }
        if (openParen.equals("=")) {
            // "a = b".
            String varName1 = relationName;
            String varName2 = nextToken(st);
            
            Variable var1 = (Variable) nameToVar.get(varName1);
            Variable var2 = (Variable) nameToVar.get(varName2);
            FieldDomain fd;
            if (var1 == null) {
                if (var2 == null) {
                    outputError(lineNum, st.getPosition(), s, "Cannot use \"=\" on two unbound variables.");
                    throw new IllegalArgumentException();
                }
                fd = var2.fieldDomain;
                var1 = parseVariable(fd, nameToVar, varName1);
            } else {
                fd = var1.fieldDomain;
                if (var2 == null) {
                    var2 = parseVariable(fd, nameToVar, varName2);
                }
            }
            if (var1.fieldDomain != var2.fieldDomain) {
                outputError(lineNum, st.getPosition(), s, "Variable "+var1+" and "+var2+" have different field domains.");
                throw new IllegalArgumentException();
            }
            
            Relation r = flip ? getNotEquivalenceRelation(fd) : getEquivalenceRelation(fd);
            List vars = new Pair(var1, var2);
            RuleTerm rt = new RuleTerm(vars, r);
            return rt;
        } else if (!openParen.equals("(")) {
            outputError(lineNum, st.getPosition(), s, "Expected \"(\" or \"=\", got \""+openParen+"\"");
            throw new IllegalArgumentException();
        }
        if (flip) {
            outputError(lineNum, st.getPosition(), s, "Unexpected \"!\"");
            throw new IllegalArgumentException();
        }
            
        Relation r = getRelation(relationName);
        if (r == null) {
            outputError(lineNum, st.getPosition(), s, "Unknown relation "+relationName);
            throw new IllegalArgumentException();
        }
        List/*<Variable>*/ vars = new LinkedList();
        for (;;) {
            if (r.fieldDomains.size() <= vars.size()) {
                outputError(lineNum, st.getPosition(), s, "Too many fields for "+r);
                throw new IllegalArgumentException();
            }
            FieldDomain fd = (FieldDomain) r.fieldDomains.get(vars.size()); 
            String varName = nextToken(st);
            Variable var = parseVariable(fd, nameToVar, varName);
            if (vars.contains(var)) {
                outputError(lineNum, st.getPosition(), s, "Duplicate variable "+var);
                throw new IllegalArgumentException();
            }
            vars.add(var);
            if (var.fieldDomain == null) var.fieldDomain = fd;
            else if (var.fieldDomain != fd) {
                outputError(lineNum, st.getPosition(), s, "Variable "+var+" used as both "+var.fieldDomain+" and "+fd);
                throw new IllegalArgumentException();
            }
            String sep = nextToken(st);
            if (sep.equals(")")) break;
            if (!sep.equals(",")) {
                outputError(lineNum, st.getPosition(), s, "Expected ',' or ')', got '"+sep+"'");
                throw new IllegalArgumentException();
            }
        }
        if (r.fieldDomains.size() != vars.size()) {
            outputError(lineNum, st.getPosition(), s, "Wrong number of vars in rule term for "+relationName);
            throw new IllegalArgumentException();
        }
        
        RuleTerm rt = new RuleTerm(vars, r);
        return rt;
    }
    
    Variable parseVariable(FieldDomain fd, Map nameToVar, String varName) {
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
        return var;
    }
    
//    public Relation getOrCreateRelation(String name, List/*<Variable>*/ vars) {
//        Relation r = (Relation) nameToRelation.get(name);
//        if (r == null) nameToRelation.put(name, r = createRelation(name, vars));
//        return r;
//    }
    
    Relation getEquivalenceRelation(FieldDomain fd) {
        Relation r = (Relation) equivalenceRelations.get(fd);
        if (r == null) {
            equivalenceRelations.put(fd, r = createEquivalenceRelation(fd));
        }
        return r;
    }
    
    Relation getNotEquivalenceRelation(FieldDomain fd) {
        Relation r = (Relation) notequivalenceRelations.get(fd);
        if (r == null) {
            notequivalenceRelations.put(fd, r = createNotEquivalenceRelation(fd));
        }
        return r;
    }
    
    void loadInitialRelations() throws IOException {
        for (Iterator i = relationsToLoad.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            try {
                r.load();
            } catch (IOException x) {
                System.out.println("WARNING: Cannot load bdd "+r+": "+x.toString());
            }
        }
        for (Iterator i = relationsToLoadTuples.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            try {
                r.loadTuples();
            } catch (IOException x) {
                System.out.println("WARNING: Cannot load tuples "+r+": "+x.toString());
            }
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
        for (Iterator i = relationsToPrintSize.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            double size = r.size();
            DecimalFormat myFormatter = new DecimalFormat("0.");
            String output = myFormatter.format(size); 
            out.println("SIZE OF "+r+": "+output);
        }
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
        for (Iterator i = relationsToDumpNegatedTuples.iterator(); i.hasNext(); ) {
            Relation r = (Relation) i.next();
            if (NOISY) out.println("Dumping negated tuples for "+r);
            r.saveNegatedTuples();
        }
    }

}
