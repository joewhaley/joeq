// GenRelations.java, created Jun 24, 2004 3:31:53 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Main;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import joeq.Compiler.Analysis.IPA.PA;
import joeq.Compiler.Quad.BasicBlockVisitor;
import joeq.Compiler.Quad.CodeCache;
import joeq.Compiler.Quad.ControlFlowGraphVisitor;
import joeq.Compiler.Quad.QuadVisitor;
import jwutil.classloader.HijackingClassLoader;
import jwutil.reflect.Reflect;

/**
 * Generate initial relations for BDD pointer analysis.
 * 
 * @author jwhaley
 * @version $Id$
 */
public class GenRelations {
    
    public static URL getFileURL(String filename) throws IOException {
        return getFileURL(null, filename);
    }
    public static URL getFileURL(String dirname, String filename) throws IOException {
        String sep = System.getProperty("file.separator");
        String name;
        if (dirname == null) name = filename;
        else if (dirname.endsWith(sep)) name = dirname+filename;
        else name = dirname+sep+filename;
        File f = new File(name);
        if (f.exists()) return f.toURL();
        else return null;
    }
    
    public static ClassLoader addBDDLibraryToClasspath(String[] args) throws IOException {
        System.out.println("BDD library is not in classpath!  Adding it.");
        String sep = System.getProperty("file.separator");
        URL url;
        url = getFileURL("joeq"+sep+"Support", "javabdd.jar");
        if (url == null)
            url = getFileURL("joeq"+sep+"Support", "javabdd_0.6.jar");
        if (url == null)
            url = getFileURL("javabdd.jar");
        if (url == null) {
            System.err.println("Cannot find JavaBDD library!");
            System.exit(-1);
            return null;
        }
        URL url2 = new File(".").toURL();
        return new HijackingClassLoader(new URL[] {url, url2});
    }
    
    public static void main(String[] args) throws IOException {
        
        // Make sure we have the BDD library in our classpath.
        try {
            Class.forName("org.sf.javabdd.BDD");
        } catch (ClassNotFoundException x) {
            ClassLoader cl = addBDDLibraryToClasspath(args);
            // Reflective invocation under the new class loader.
            Reflect.invoke(cl, GenRelations.class.getName(), "main2", new Class[] {String[].class}, new Object[] {args});
            return;
        }
        
        // Just call it directly.
        main2(args);
    }
    
    public static void main2(String[] args) throws IOException {
        
        if (args.length == 0) {
            printUsage();
            return;
        }
        
        boolean CS = false;
        boolean FLY = false;
        boolean SSA = false;
        
        int i;
        for (i = 0; i < args.length; ++i) {
            if (args[i].equalsIgnoreCase("addpass")) {
                String passname = args[++i];
                ControlFlowGraphVisitor mv = null;
                BasicBlockVisitor bbv = null;
                QuadVisitor qv = null;
                Object o;
                try {
                    Class c = Class.forName(passname);
                    o = c.newInstance();
                    if (o instanceof ControlFlowGraphVisitor) {
                        mv = (ControlFlowGraphVisitor) o;
                    } else {
                        if (o instanceof BasicBlockVisitor) {
                            bbv = (BasicBlockVisitor) o;
                        } else {
                            if (o instanceof QuadVisitor) {
                                qv = (QuadVisitor) o;
                            } else {
                                System.err.println("Unknown pass type " + c);
                            }
                            bbv = new QuadVisitor.AllQuadVisitor(qv, false);
                        }
                        mv = new BasicBlockVisitor.AllBasicBlockVisitor(bbv, false);
                    }
                    CodeCache.passes.add(mv);
                } catch (java.lang.ClassNotFoundException x) {
                    System.err.println("Cannot find pass named " + passname + ".");
                    System.err.println("Check your classpath and make sure you compiled your pass.");
                } catch (java.lang.InstantiationException x) {
                    System.err.println("Cannot instantiate pass " + passname + ": " + x);
                } catch (java.lang.IllegalAccessException x) {
                    System.err.println("Cannot access pass " + passname + ": " + x);
                    System.err.println("Be sure that you made your class public?");
                }
            } else if (args[i].equals("-cs")) CS = true;
            else if (args[i].equals("-fly")) FLY = true;
            else if (args[i].equals("-ssa")) SSA = true;
            else break;
        }
        if (i > 0) {
            String[] args2 = new String[args.length - i];
            System.arraycopy(args, i, args2, 0, args2.length);
            args = args2;
        }
        
        System.setProperty("pa.skipsolve", "yes");
        System.setProperty("pa.dumpinitial", "yes");
        System.setProperty("pa.dumpresults", "no");
        if (CS) System.setProperty("pa.cs", "yes");
        if (FLY) System.setProperty("pa.dumpfly", "yes");
        if (SSA) System.setProperty("pa.dumpssa", "yes");
        String dumppath = System.getProperty("pa.dumppath");
        if (dumppath != null) {
            if (dumppath.length() > 0) {
                File f = new File(dumppath);
                if (!f.exists()) f.mkdirs();
                String sep = System.getProperty("file.separator");
                if (!dumppath.endsWith(sep)) dumppath += sep;
            }
            if (System.getProperty("pa.callgraph") == null) {
                System.setProperty("pa.callgraph", dumppath+"callgraph");
            }
        }
        
        PA.main(args);
    }
    
    public static void printUsage() {
        System.out.println("Usage: java "+GenRelations.class.getName()+" <options> <class> (<method>)");
        System.out.println("Usage: java "+GenRelations.class.getName()+" <options> @<classlist>");
        System.out.println("Valid options:");
        System.out.println(" -cs     context-sensitive");
        System.out.println(" -fly    on-the-fly call graph");
        System.out.println(" -ssa    also dump SSA representation");
        System.out.println("Other system properties:");
        System.out.println(" -Dpa.dumppath      where to save the relations");
        System.out.println(" -Dpa.icallgraph    location to load initial call graph, blank to force callgraph regeneration");
        System.out.println(" -Dpa.dumpdotgraph  dump the call graph in dot graph format");
    }
    
}