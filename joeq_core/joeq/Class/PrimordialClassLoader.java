/*
 * BootstrapClassLoader.java
 *
 * Created on January 15, 2001, 12:59 AM
 *
 * @author  jwhaley
 * @version 
 */

package Bootstrap;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.PrintStream;
import java.io.IOException;

import java.util.Set;
import java.util.List;
import java.util.Iterator;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import java.util.zip.ZipFile;
import java.util.zip.ZipEntry;

import jq;
import Clazz.*;
import UTF.Utf8;
import Util.ArrayIterator;
import Util.EnumerationIterator;
import Util.FilterIterator;
import Util.UnmodifiableIterator;
import Util.Default;

public class PrimordialClassLoader extends ClassLoader implements jq_ClassFileConstants {
    
    public static /*final*/ boolean TRACE = false;
    public static final PrintStream out = System.out;
    
    static abstract class ClasspathElement {
        /** Open a stream to read the given resource, or return
         *  <code>null</code> if resource cannot be found. */
        abstract InputStream getResourceAsStream(String resourcename);
        /** Iterate over all classes in the given package. */
        abstract Iterator listPackage(String packagename);
    }
    /** A .zip or .jar file in the CLASSPATH. */
    static class ZipFileElement extends ClasspathElement {
        ZipFile zf;
        ZipFileElement(ZipFile zf) { this.zf = zf; }
        public String toString() { return zf.getName(); }
        InputStream getResourceAsStream(String name) {
            try { // look for name in zipfile, return null if something goes wrong.
                ZipEntry ze = zf.getEntry(name);
                return (ze==null)?null:zf.getInputStream(ze);
            } catch (UnsatisfiedLinkError e) {
                System.err.println("UNSATISFIED LINK ERROR: "+name);
                return null;
            } catch (IOException e) { return null; }
        }
        Iterator listPackage(final String pathname) {
            // look for directory name first
            final String filesep   = System.getProperty("file.separator");
            /* not all .JAR files have entries for directories, unfortunately.
            ZipEntry ze = zf.getEntry(pathname);
            if (ze==null) return Default.nullIterator;
             */
            return new FilterIterator(new EnumerationIterator(zf.entries()),
            new FilterIterator.Filter() {
                public boolean isElement(Object o) { ZipEntry zze=(ZipEntry) o;
                    String name = zze.getName();
                    return (!zze.isDirectory()) && name.startsWith(pathname) &&
                    name.lastIndexOf(filesep)==(pathname.length()-1);
                }
                public Object map(Object o) {
                    return ((ZipEntry)o).getName();
                }
            });
        }
        /** Close the zipfile when this object is garbage-collected. */
        protected void finalize() throws Throwable {
            // yes, it is possible to finalize an uninitialized object.
            try { if (zf!=null) zf.close(); } finally { super.finalize(); }
        }
    }
    /** A regular path string in the CLASSPATH. */
    static class PathElement extends ClasspathElement {
        String path;

        PathElement(String path) { this.path = path; }
        public String toString() { return path; }
        InputStream getResourceAsStream(String name) {
            try { // try to open the file, starting from path.
                final String filesep = System.getProperty("file.separator");
                if (filesep.charAt(0) != '/') name = name.replace('/', filesep.charAt(0));
                File f = new File(path, name);
                return new FileInputStream(f);
            } catch (FileNotFoundException e) {
                return null; // if anything goes wrong, return null.
            }
        }
        Iterator listPackage(final String pathname) {
            File f = new File(path,pathname);
            if (!f.exists() || !f.isDirectory()) return Default.nullIterator;
            return new FilterIterator(new ArrayIterator(f.list()),
                new FilterIterator.Filter() {
                    public Object map(Object o) { return pathname + ((String)o); }
                });
        }
    }

    /** Vector of ClasspathElements corresponding to CLASSPATH entries. */
    public void initClasspath(String s) { // initialize classpathVector.
        Set duplicates = new HashSet(); // don't add duplicates.
        for (Iterator it = classpaths(s); it.hasNext(); ) {
            String path = (String) it.next();
            if (duplicates.contains(path)) continue; // skip duplicate.
            else duplicates.add(path);
            if (path.toLowerCase().endsWith(".zip") ||
                path.toLowerCase().endsWith(".jar"))
                try {
                    classpathList.add(new ZipFileElement(new ZipFile(path)));
                } catch (IOException ex) { /* skip this zip file, then. */ }
            else {
                classpathList.add(new PathElement(path));
            }
        }
        ((ArrayList) classpathList).trimToSize(); // save memory.
    }

    /** Iterate over the components of the system CLASSPATH.
     *  Each element is a <code>String</code> naming one segment of the
     *  CLASSPATH. */
    public static final Iterator classpaths(String classpath) {

        final String pathsep = System.getProperty("path.separator");
        
        // For convenience, make sure classpath begins with and ends with pathsep.
        if (!classpath.startsWith(pathsep)) classpath = pathsep + classpath;
        if (!classpath.endsWith(pathsep)) classpath = classpath + pathsep;
        final String cp = classpath;

        return new UnmodifiableIterator() {
            int i=0;
            public boolean hasNext() {
                return (cp.length() > (i+pathsep.length()));
            }
            public Object next() {
                i+=pathsep.length(); // cp begins with pathsep.
                String path = cp.substring(i, cp.indexOf(pathsep, i));
                i+=path.length(); // skip over path.
                return path;
            }
        };
    }

    public String classpathToString() {
        final String pathsep = System.getProperty("path.separator");
        StringBuffer result = new StringBuffer(pathsep);
        for (Iterator it = classpathList.iterator(); it.hasNext(); ) {
            ClasspathElement cpe = (ClasspathElement) it.next();
            result.append(cpe.toString());
            result.append(pathsep);
        }
        return result.toString();
    }
    
    public static String descriptorToResource(String desc) {
        jq.assert(desc.charAt(0)==TC_CLASS);
        jq.assert(desc.charAt(desc.length()-1)==TC_CLASSEND);
        jq.assert(desc.indexOf('.')==-1); // should have '/' separators.
        return desc.substring(1, desc.length()-1) + ".class";
    }
    
    /** Translate a class name into a corresponding resource name.
     * @param classname The class name to translate.
     */
    public static String classnameToResource(String classname) {
        jq.assert(classname.indexOf('/')==-1); // should have '.' separators.
        String filesep = System.getProperty("file.separator");
        // Swap all '.' for '/' & append ".class"
        return classname.replace('.', filesep.charAt(0)) + ".class";
    }

    /** Open an <code>InputStream</code> on a resource found somewhere
     *  in the CLASSPATH.
     * @param name The filename of the resource to locate.
     */
    public InputStream getResourceAsStream(String name) {
        //if (jq.Bootstrapping && name.startsWith("java/")) {
        //    // hijack loading of java/* to point to bootstrap versions
        //    char[] c = name.toCharArray();
        //    c[3] = '_';
        //    String name2 = new String(c);
        //    for (Iterator it = classpathList.iterator(); it.hasNext(); ) {
        //        ClasspathElement cpe = (ClasspathElement) it.next();
        //        InputStream is = cpe.getResourceAsStream(name2);
        //        if (is!=null) {
        //            return is; // return stream if found.
        //        }
        //    }
        //}
        for (Iterator it = classpathList.iterator(); it.hasNext(); ) {
            ClasspathElement cpe = (ClasspathElement) it.next();
            InputStream is = cpe.getResourceAsStream(name);
            if (is!=null) {
                return is; // return stream if found.
            }
        }
        // Couldn't find resource.
        return null;
    }
    
    private final List/*<ClasspathElement>*/ classpathList;
    
    public static final PrimordialClassLoader loader;
    static {
        loader = new PrimordialClassLoader();
        initPrimitiveTypes();
    }
    
    private PrimordialClassLoader() {
        bs_desc2type = new HashMap();
        classpathList = new ArrayList();
    }
    
    private static void initPrimitiveTypes() {
        // trigger jq_Primitive clinit
        loader.getOrCreateBSType(jq_Primitive.BYTE.getDesc());
        loader.bs_desc2type.put(jq_Array.BYTE_ARRAY.getDesc(), jq_Array.BYTE_ARRAY);
        loader.bs_desc2type.put(jq_Array.CHAR_ARRAY.getDesc(), jq_Array.CHAR_ARRAY);
        loader.bs_desc2type.put(jq_Array.DOUBLE_ARRAY.getDesc(), jq_Array.DOUBLE_ARRAY);
        loader.bs_desc2type.put(jq_Array.FLOAT_ARRAY.getDesc(), jq_Array.FLOAT_ARRAY);
        loader.bs_desc2type.put(jq_Array.INT_ARRAY.getDesc(), jq_Array.INT_ARRAY);
        loader.bs_desc2type.put(jq_Array.LONG_ARRAY.getDesc(), jq_Array.LONG_ARRAY);
        loader.bs_desc2type.put(jq_Array.SHORT_ARRAY.getDesc(), jq_Array.SHORT_ARRAY);
        loader.bs_desc2type.put(jq_Array.BOOLEAN_ARRAY.getDesc(), jq_Array.BOOLEAN_ARRAY);
    }
    
    public DataInputStream getClassFileStream(Utf8 descriptor)
    throws IOException {
        String resourceName = descriptorToResource(descriptor.toString());
        InputStream is = getResourceAsStream(resourceName);
        if (is == null) return null;
        return new DataInputStream(is);
    }

    public jq_Class getJavaLangObject() {
        //return (jq_Class)getOrCreateBSType("Ljava/lang/Object;");
        return ClassLib.sun13.java.lang.Object._class;
    }
    public jq_Class getJavaLangString() {
        //return (jq_Class)getOrCreateBSType("Ljava/lang/String;");
        return ClassLib.sun13.java.lang.String._class;
    }
    public jq_Class getJavaLangSystem() {
        //return (jq_Class)getOrCreateBSType("Ljava/lang/System;");
        return ClassLib.sun13.java.lang.System._class;
    }
    public jq_Class getJavaLangThrowable() {
        //return (jq_Class)getOrCreateBSType("Ljava/lang/Throwable;");
        return ClassLib.sun13.java.lang.Throwable._class;
    }

    private final Map/*<Utf8, jq_Type>*/ bs_desc2type;

    public Set/*jq_Type*/ getAllTypes() {
        Iterator i = bs_desc2type.entrySet().iterator();
        HashSet s = new HashSet();
        while (i.hasNext()) {
            Map.Entry e = (Map.Entry)i.next();
            s.add(e.getValue());
        }
        return s;
    }
    
    public final jq_Type getOrCreateBSType(String desc) { return getOrCreateBSType(Utf8.get(desc)); }
    public final jq_Type getOrCreateBSType(Utf8 desc) {
        if (!jq.Bootstrapping) return ClassLib.sun13.java.lang.ClassLoader.getOrCreateType(this, desc);
        jq_Type t = (jq_Type)bs_desc2type.get(desc);
        if (t == null) {
            if (desc.isDescriptor(jq_ClassFileConstants.TC_CLASS)) {
                if (TRACE) out.println("Adding class type "+desc);
                bs_desc2type.put(desc, t = jq_Class.newClass(this, desc));
            } else if (desc.isDescriptor(jq_ClassFileConstants.TC_ARRAY)) {
                if (TRACE) out.println("Adding array type "+desc);
                Utf8 elementDesc = desc.getArrayElementDescriptor();
                jq_Type elementType = getOrCreateBSType(elementDesc); // recursion
                bs_desc2type.put(desc, t = jq_Array.newArray(desc, this, elementType));
            } else {
                // this code only gets executed at the very beginning, when creating primitive types.
                if (desc == Utf8.get((char)TC_BYTE+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "byte", 1));
                else if (desc == Utf8.get((char)TC_CHAR+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "char", 2));
                else if (desc == Utf8.get((char)TC_DOUBLE+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "double", 8));
                else if (desc == Utf8.get((char)TC_FLOAT+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "float", 4));
                else if (desc == Utf8.get((char)TC_INT+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "int", 4));
                else if (desc == Utf8.get((char)TC_LONG+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "long", 8));
                else if (desc == Utf8.get((char)TC_SHORT+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "short", 2));
                else if (desc == Utf8.get((char)TC_BOOLEAN+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "boolean", 1));
                else if (desc == Utf8.get((char)TC_VOID+""))
                    bs_desc2type.put(desc, t = jq_Primitive.newPrimitive(desc, "void", 0));
                /*
                else if (desc == jq_Array.BYTE_ARRAY.getDesc()) return jq_Array.BYTE_ARRAY;
                else if (desc == jq_Array.CHAR_ARRAY.getDesc()) return jq_Array.CHAR_ARRAY;
                else if (desc == jq_Array.DOUBLE_ARRAY.getDesc()) return jq_Array.DOUBLE_ARRAY;
                else if (desc == jq_Array.FLOAT_ARRAY.getDesc()) return jq_Array.FLOAT_ARRAY;
                else if (desc == jq_Array.INT_ARRAY.getDesc()) return jq_Array.INT_ARRAY;
                else if (desc == jq_Array.LONG_ARRAY.getDesc()) return jq_Array.LONG_ARRAY;
                else if (desc == jq_Array.SHORT_ARRAY.getDesc()) return jq_Array.SHORT_ARRAY;
                else if (desc == jq_Array.BOOLEAN_ARRAY.getDesc()) return jq_Array.BOOLEAN_ARRAY;
                 */
                else jq.UNREACHABLE("bad descriptor! "+desc);
            }
        }
        return t;
    }
    
    public void unloadBSType(jq_Type t) {
        bs_desc2type.remove(t.getDesc());
    }
}
