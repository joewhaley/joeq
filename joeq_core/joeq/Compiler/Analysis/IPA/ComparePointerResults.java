/*
 * Created on Oct 4, 2003
 *
 * To change the template for this generated file go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
package Compil3r.Analysis.IPA;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import org.sf.javabdd.BDDFactory;

/**
 * @author jwhaley
 *
 * To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
public class ComparePointerResults {

    static Collection/*<CSPAResults>*/ results;
    static BDDFactory bdd;
    
    public static void main(String[] args) throws IOException {
        if (args.length == 0)
            args = new String[] { "pa", "cspa" };
        
        bdd = CSPAResults.initialize((String) null);
        CSPAResults r = new CSPAResults(bdd);
        r.loadCallGraph("callgraph");
        
        results = new ArrayList(args.length);
        for (int i = 0; i < args.length; ++i) {
            String name = args[i];
            System.out.println("Loading \""+name+"\" results...");
            CSPAResults r2 = new CSPAResults(bdd);
            r2.cg = r.cg; r2.pn = r.pn;
            r2.load(name);
            r2.reindex(r);
            results.add(r2);
            r = r2;
        }
        System.out.println("Done loading results.");
    }
    
    public static void compareEscapeAnalysis() {
        for (Iterator i = results.iterator(); i.hasNext(); ) {
            CSPAResults r = (CSPAResults) i.next();
            r.escapeAnalysis();
        }
    }
}
