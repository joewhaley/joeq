package joeq.Compiler.Analysis.FlowInsensitive;

import java.beans.PrimitivePersistenceDelegate;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.StringTokenizer;
import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
import joeq.Compiler.Analysis.IPA.PA;
import joeq.Compiler.Analysis.IPA.ProgramLocation;
import joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import joeq.Compiler.Quad.CodeCache;
import joeq.Main.HostedVM;
import joeq.UTF.Utf8;
import jwutil.util.Assert;

/**
 * @author V.Benjamin Livshits
 * @version $Id$
 * 
 * This class declares methods for resolving reflective calls.
 */
public abstract class ReflectionInformationProvider {
    public class NewInstanceTargets {
        private jq_Method declaredIn;
        private Collection targets = new LinkedList();

        /**
         * @param declaredIn
         */
        public NewInstanceTargets(String declaredIn) {
            this.declaredIn = getMethod(declaredIn);
            if(PA.TRACE_REFLECTION && this.declaredIn == null) {
                System.out.println("No method for " + declaredIn + " in NewInstanceTargets. "
                    + " The classpath is [" + PrimordialClassLoader.loader.classpathToString() + "]");
            }
        }
        
        public boolean isValid(){
            return 
                getDeclaredIn() != null &&
                targets.size() > 0;
        }

        private jq_Method getMethod(String fullMethodName) {
            int index = fullMethodName.lastIndexOf('.');
            Assert._assert(index != -1);
            
            String className = fullMethodName.substring(0, index);
            String methodName = fullMethodName.substring(index+1, fullMethodName.length());
            
            String classdesc = "L" + className.replace('.', '/') + ";";
            jq_Class clazz = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(classdesc);
            
//            jq_Class clazz = (jq_Class) jq_Type.parseType(className);
            try {
                clazz.prepare();
            } catch (NoClassDefFoundError e){
                // no class found
                return null;
            }
            jq_Method method = clazz.getDeclaredMethod(methodName);
            Assert._assert(method != null);
            
            return method;
        }
        
        private jq_Class getClass(String className) {            
            jq_Class clazz = (jq_Class) jq_Type.parseType(className);
            try {
                clazz.prepare();
            } catch (NoClassDefFoundError e){
                return null;
            }
            Assert._assert(clazz != null);
            
            return clazz;
        }

        public void addTarget(String target) {
            jq_Class clazz = getClass(target);
            if(clazz != null){
                addTarget(clazz);
            }
        }

        public void addTarget(jq_Class clazz) {
            jq_Method constructor = clazz.getInitializer(Utf8.get("()V"));
            targets.add(constructor);
        }
        
        public String toString(){
            return declaredIn + " -> " + targets.toString();            
        }

        public jq_Method getDeclaredIn() {
            return this.declaredIn;
        }

        public Collection getTargets() {
            return this.targets;
        }
        
        public void addSubclasses(String className) {
            jq_Class clazz = getClass(className);
            Assert._assert(false);  
        }        
    }
    
    /** 
     * Reflective methods to be used in isReflective(...)
     * 
     * @see isReflective 
     * */
    private static String[][] methodSpecs = { 
                            {"java.lang.Class", "forName"},
                            {"java.lang.Object", "newInstance"},
                            {"java.lang.reflection.Constructor", "newInstance"},
                            };
    
    /**
     * Checks if method is reflective.
     * 
     * @param method
     */
    public static boolean isReflective(jq_Method method){
        for(int i = 0; i < methodSpecs.length; i++){
            String[] methodSpec = methodSpecs[i];
            String className = methodSpec[0];
            String methodName = methodSpec[1];
            
            if(!className.equals(method.getDeclaringClass().getName())) continue;
            if(!methodName.toString().equals(method.getName())) continue;
            
            return true;
        }
        
        return false;
    }

    /**
     * Checks if mc corresponds to a newInstance call.
     */
    public static boolean isNewInstance(ProgramLocation.QuadProgramLocation mc){
        jq_Method target = mc.getTargetMethod();
        return isNewInstance(target);
    }

    /**
     * Checks if target is a newInstance method. 
     */
    public static boolean isNewInstance(jq_Method target) {
        String className = target.getDeclaringClass().getName(); 
        String methodName = target.getName().toString();
        
        if(!className.equals("java.lang.Class")) return false;
        if(!methodName.equals("newInstance")) return false;
        
        return true;
    }

    /**
     * Resolves constructors being pointed to by a newInstance() call mc.
     * */
    public abstract Collection/*<jq_Method>*/  getNewInstanceTargets(ProgramLocation.QuadProgramLocation mc);
    
    /**
     * Resolves constructors being pointed to by a newInstance() calls within 
     * method n.
     * 
     * Notice that information may be imprecise because we only have one piece of 
     * data per method.
     * */
    public abstract Collection/*<jq_Method>*/  getNewInstanceTargets(jq_Method n);
    
    /**
     * This implementation of ReflectionInformationProvider 
     * reads answers from a file. 
     * */
    public static class CribSheetReflectionInformationProvider extends ReflectionInformationProvider {
        private static boolean TRACE = true;
        private static final String DEFAULT_CRIB_FILE = "reflection.spec";

        public CribSheetReflectionInformationProvider(String cribSheetFileName){
            try {
                readSpec(cribSheetFileName);
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
              
        public CribSheetReflectionInformationProvider() {
            this(DEFAULT_CRIB_FILE);
        }

        public static void main(String[] args) {
            HostedVM.initialize();
            CodeCache.AlwaysMap = true;
            TRACE = true;
            
            CribSheetReflectionInformationProvider provider = 
                new CribSheetReflectionInformationProvider(args[0]);
        }
        
        /**
         * @param cribSheetFileName
         * @throws IOException
         */
        private void readSpec(String cribSheetFileName) throws IOException {
            FileReader fileIn = new FileReader(cribSheetFileName);
            LineNumberReader in = new LineNumberReader(fileIn);
            String line = in.readLine();
            do {
                if(!line.startsWith("#") && line.trim().length() > 0){
                    NewInstanceTargets spec = parseSpecLine(line);
                    if(spec.isValid()){
                        if(PA.TRACE_REFLECTION){
                            System.out.println("Adding a reflection spec for " + spec.getDeclaredIn());
                        }
                        specs.add(spec);
                    }
                }
                line = in.readLine();
            } while (line != null);
            in.close();
        }
        
        Collection/*<NewInstanceTargets>*/ specs      = new LinkedList();
        private static final Object SUBCLASSES_MARKER = "<";
        private static final Object ELLIPSES          = "...";

        /**
         * Parses one line like this:
             org.roller.presentation.RollerContext.getAuthenticator org.roller.presentation.DefaultAuthenticator ...
        */
        private NewInstanceTargets parseSpecLine(String line) {
            StringTokenizer tok = new StringTokenizer(line);
            String declaredIn = tok.nextToken();
            NewInstanceTargets targets = new NewInstanceTargets(declaredIn);
            while(tok.hasMoreTokens()){
                String token = tok.nextToken();
                if(!token.equals(ELLIPSES)){
                    if(!token.equals(SUBCLASSES_MARKER)){
                        targets.addTarget(token);
                    }else{
                        targets.addSubclasses(tok.nextToken());
                    }
                }else{
                    System.err.println("Specification for " + declaredIn + " is incomplete.");
                }
            }
            if(TRACE && targets.isValid()){
                System.out.println("Read " + targets);
            }
            
            return targets;
        }
        
        /* (non-Javadoc)
         * @see joeq.Compiler.Analysis.FlowInsensitive.ReflectionInformationProvider#getNewInstanceTargets(joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation)
         */
        public Collection getNewInstanceTargets(QuadProgramLocation mc) {
            // TODO
            return null;
        }
        
        public Collection/*<jq_Method>*/ getNewInstanceTargets(jq_Method n) {
            for(Iterator iter = specs.iterator(); iter.hasNext();){
                NewInstanceTargets spec = (NewInstanceTargets) iter.next();
                
                if(spec.getDeclaredIn() == n){
                    return spec.getTargets();
                }
            }
            if(PA.TRACE_REFLECTION){
                System.out.println("No information for method " + n);
            }
            return null;            
        }
    }
}
