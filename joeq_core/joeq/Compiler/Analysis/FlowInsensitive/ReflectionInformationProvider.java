package joeq.Compiler.Analysis.FlowInsensitive;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.StringTokenizer;
import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
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
    /**
     * @author Vladimir Livshits
     * 
     */
    public class NewInstanceTargets {

        private jq_Method declaredIn;
        private Collection targets = new LinkedList();

        /**
         * @param declaredIn
         */
        public NewInstanceTargets(String declaredIn) {
            this.declaredIn = getMethod(declaredIn);
        }

        private jq_Method getMethod(String fullMethodName) {
            int index = fullMethodName.lastIndexOf('.');
            Assert._assert(index != -1);
            
            String className = fullMethodName.substring(0, index);
            String methodName = fullMethodName.substring(index+1, fullMethodName.length());
//            className = "L" + className + ";";
            
            jq_Class clazz = (jq_Class) jq_Type.parseType(className);
            clazz.prepare();
            jq_Method method = clazz.getDeclaredMethod(methodName);
            Assert._assert(method != null);
            
            return method;
        }
        
        private jq_Class getClass(String className) {
            jq_Class clazz = (jq_Class) jq_Type.parseType(className);
            clazz.prepare();
            Assert._assert(clazz != null);
            
            return clazz;
        }

        public void addTarget(String target) {
            addTarget(getClass(target));            
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
        
        return className.equals("java.lang.Class") && methodName.equals("newInstance");
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
        private static final boolean TRACE = true;
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
                NewInstanceTargets spec = parseSpecLine(line);
                specs.add(spec);
                line = in.readLine();
            } while (line != null);
            in.close();
        }
        
        Collection/*<NewInstanceTargets>*/ specs = new LinkedList();
        
        private NewInstanceTargets parseSpecLine(String line) {
            //org.roller.presentation.RollerContext.getAuthenticator org.roller.presentation.DefaultAuthenticator ...
            StringTokenizer tok = new StringTokenizer(line);
            String declaredIn = tok.nextToken();
            NewInstanceTargets targets = new NewInstanceTargets(declaredIn);
            while(tok.hasMoreElements()){
                targets.addTarget(tok.nextToken());
            }
            if(TRACE) System.out.println("Read " + targets);
            
            return targets;
        }
        
        /* (non-Javadoc)
         * @see joeq.Compiler.Analysis.FlowInsensitive.ReflectionInformationProvider#getNewInstanceTargets(joeq.Compiler.Analysis.IPA.ProgramLocation.QuadProgramLocation)
         */
        public Collection getNewInstanceTargets(QuadProgramLocation mc) {            
            return null;
        }
        
        public Collection/*<jq_Method>*/ getNewInstanceTargets(jq_Method n) {
            for(Iterator iter = specs.iterator(); iter.hasNext();){
                NewInstanceTargets spec = (NewInstanceTargets) iter.next();
                
                if(spec.getDeclaredIn() == n){
                    return spec.getTargets();
                }
            }
            return null;            
        }
    }
}
