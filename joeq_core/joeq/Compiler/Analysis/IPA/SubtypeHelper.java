package joeq.Compiler.Analysis.IPA;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_Reference;
import joeq.Class.jq_Reference.jq_NullType;
import net.sf.javabdd.BDD;

public class SubtypeHelper {
    private PA pa;
    private Map/*<jq_Class, Collection<jq_Class>>*/ type2subtypeCache = new HashMap();
    private boolean TRACE = true; 

    public SubtypeHelper(PA pa){
        this.pa = pa;
    }

    public Collection getKnownSubtypes(jq_Class t) {
        Collection result = new LinkedList();
        int T1_i = pa.Tmap.get(t);
        BDD subtypes = pa.aT.relprod(pa.T1.ithVar(T1_i), pa.T1set);          // T2
        for(Iterator typeIter = subtypes.iterator(pa.T2set); typeIter.hasNext();){
            jq_Reference subtype = (jq_Reference) pa.Tmap.get(((BDD)typeIter.next()).scanVar(pa.T2).intValue());
            if (subtype == null || subtype == jq_NullType.NULL_TYPE) continue;
            if(!(subtype instanceof jq_Class)){
                System.err.println("Skipping a non-class type: " + t);
                continue;
            }
            jq_Class c = (jq_Class) subtype;
            result.add(c);
        }
        
        return result;
    }
    
    static String canonicalizeClassName(String s) {
        if (s.endsWith(".class")) s = s.substring(0, s.length() - 6);
        s = s.replace('.', '/');
        String desc = "L" + s + ";";
        return desc;
    }
    
    public Collection getAllSubtypes(jq_Class clazz) {
        Collection result = (Collection) type2subtypeCache.get(clazz);
        if(result != null) {
            return result;
        }
        result = new LinkedList();
        for(Iterator iter = PrimordialClassLoader.loader.listPackages(); iter.hasNext();){
            //System.out.println("\t" + iter.next());
            String packageName = (String) iter.next();
            HashSet loaded = new HashSet();
            if(TRACE) System.out.println("Processing package " + packageName);
            
            for(Iterator classIter = PrimordialClassLoader.loader.listPackage(packageName, true); classIter.hasNext();){
                String className = (String) classIter.next();
                String canonicalClassName = canonicalizeClassName(className);
                if (loaded.contains(canonicalClassName))
                    continue;
                loaded.add(canonicalClassName);
                try {
                    jq_Class c = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType(canonicalClassName);
                    c.load();
                    c.prepare();
                    if(c.isSubtypeOf(clazz)){                        
                        System.out.println("Initialized a subclass of " + clazz + ", class: " + c);
                        result.add(c);
                    }
                } catch (NoClassDefFoundError x) {
                    if(TRACE) System.err.println("Package " + packageName + ": Class not found (canonical name " + canonicalClassName + ").");
                } catch (LinkageError le) {
                    if(TRACE) System.err.println("Linkage error occurred while loading class (" + canonicalClassName + "):" + le.getMessage());
                    //le.printStackTrace(System.err);
                } catch (RuntimeException e){
                    if(TRACE) System.err.println("Security error occured: " + e.getMessage());
                }
            }            
        }
     
        type2subtypeCache.put(clazz, result);
        return result;   
    }

    public Collection getSubtypes(jq_Class clazz, boolean use_known_subtypes_for_reflection) {
        if(use_known_subtypes_for_reflection)
            return getKnownSubtypes(clazz);
        else
            return getAllSubtypes(clazz);
    }
}
