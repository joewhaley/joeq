// GenRelations.java, created Jun 24, 2004 3:31:53 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Main;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import joeq.Compiler.Analysis.IPA.PA;
import jwutil.classloader.HijackingClassLoader;

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
    
    public static Object invoke(ClassLoader cl, String className,
        String methodName, Class[] argTypes, Object[] args) {
        Class c;
        try {
            c = Class.forName(className, true, cl);
        } catch (ClassNotFoundException e0) {
            System.err.println("Cannot load "+className);
            e0.printStackTrace();
            return null;
        }
        Method m;
        try {
            if (argTypes != null) {
                m = c.getMethod(methodName, argTypes);
            } else {
                m = null;
                Method[] ms = c.getDeclaredMethods();
                for (int i = 0; i < ms.length; ++i) {
                    if (ms[i].getName().equals(methodName)) {
                        m = ms[i];
                        break;
                    }
                }
                if (m == null) {
                    System.err.println("Can't find "+className+"."+methodName);
                    return null;
                }
            }
            m.setAccessible(true);
        } catch (SecurityException e1) {
            System.err.println("Cannot access "+className+"."+methodName);
            e1.printStackTrace();
            return null;
        } catch (NoSuchMethodException e1) {
            System.err.println("Can't find "+className+"."+methodName);
            e1.printStackTrace();
            return null;
        }
        Object result;
        try {
            result = m.invoke(null, args);
        } catch (IllegalArgumentException e2) {
            System.err.println("Illegal argument exception");
            e2.printStackTrace();
            return null;
        } catch (IllegalAccessException e2) {
            System.err.println("Illegal access exception");
            e2.printStackTrace();
            return null;
        } catch (InvocationTargetException e2) {
            if (e2.getCause() instanceof RuntimeException)
                throw (RuntimeException) e2.getCause();
            if (e2.getCause() instanceof Error)
                throw (Error) e2.getCause();
            System.err.println("Unexpected exception thrown!");
            e2.getCause().printStackTrace();
            return null;
        }
        return result;
    }
    
    public static void main(String[] args) throws IOException {
        
        // Make sure we have the BDD library in our classpath.
        try {
            Class.forName("org.sf.javabdd.BDD");
        } catch (ClassNotFoundException x) {
            ClassLoader cl = addBDDLibraryToClasspath(args);
            // Reflective invocation under the new class loader.
            invoke(cl, GenRelations.class.getName(), "main2", new Class[] {String[].class}, new Object[] {args});
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
            if (args[i].equals("-cs")) CS = true;
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