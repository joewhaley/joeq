// ResidentJava.java, created May 26, 2004 6:31:55 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Main;

import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import joeq.Util.IO.MyStringTokenizer;

/**
 * ResidentJava
 * 
 * @author jwhaley
 * @version $Id$
 */
public class ResidentJava {
    
    public static class SystemExitException extends SecurityException {

        int status;
        
        /**
         * @param status
         */
        public SystemExitException(int status) {
            this.status = status;
        }
        
    }
    
    public static void trapOnSystemExit() {
        SecurityManager sm = new SecurityManager() {
            public void checkAccept(String host, int port) {}
            public void checkAccess(Thread t) {}
            public void checkAccess(ThreadGroup t) {}
            public void checkAwtEventQueueAccess(ThreadGroup t) {}
            public void checkConnect(String host, int port) {}
            public void checkConnect(String host, int port, Object context) {}
            public void checkCreateClassLoader() {}
            public void checkDelete() {}
            public void checkExec(String file) {}
            public void checkExit(int status) {
                throw new SystemExitException(status);
            }
            public void checkLink(String lib) {}
            public void checkListen(int port) {}
            public void checkMemberAccess(Class clazzz, int which) {}
            public void checkMulticast(java.net.InetAddress maddr) {}
            public void checkPackageAccess(String pkg) {}
            public void checkPackageDefinition(String pkg) {}
            public void checkPermission(java.security.Permission perm) {}
            public void checkPermission(java.security.Permission perm, Object context) {}
            public void checkPrintJobAccess() {}
            public void checkPropertiesAccess() {}
            public void checkPropertyAccess(String key) {}
            public void checkRead(java.io.FileDescriptor fd) {}
            public void checkRead(String file) {}
            public void checkRead(String file, Object context) {}
            public void checkSecurityAccess(String target) {}
            public void checkSetFactory() {}
            public void checkSystemClipboardAccess() {}
            public boolean checkTopLevelWindow(Object window) { return true; }
            public void checkWrite(java.io.FileDescriptor fd) {}
            public void checkWrite(String file) {}
        };
        System.setSecurityManager(sm);
    }
    
    public static void main(String[] args) throws IOException {
        trapOnSystemExit();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        for (;;) {
            String commandLine = in.readLine();
            if (commandLine == null) break;
            executeProgram(commandLine);
        }
    }
    
    public static void executeProgram(String line) {
        ExecuteOptions ops;
        try {
            ops = new ExecuteOptions(line);
            try {
                ops.go();
            } catch (InvocationTargetException e) {
                Throwable x = e.getCause();
                if (x instanceof SystemExitException) {
                    int status = ((SystemExitException)x).status;
                    if (status != 0) {
                        System.err.println("Java process exited with error code "+status);
                    }
                } else {
                    System.err.println("Java process ended with exception: "+x.toString());
                    x.printStackTrace(System.err);
                }
            }
        } catch (SecurityException e) {
            System.err.println("Security exception while accessing class/method.");
        } catch (IllegalAccessException e) {
            System.err.println("Class/method is not public.");
        } catch (ClassNotFoundException e) {
            System.err.println("Class not found.");
        } catch (NoSuchMethodException e) {
            System.err.println("Class does not contain an appropriate main method.");
        }
    }
    
    public static class ExecuteOptions {
        
        Properties properties;
        Method mainMethod;
        String[] args;
        
        ExecuteOptions(String commandLine) throws ClassNotFoundException, SecurityException, NoSuchMethodException {
            properties = new Properties(System.getProperties());
            MyStringTokenizer st = new MyStringTokenizer(commandLine);
            ClassLoader myClassLoader = ClassLoader.getSystemClassLoader();
            while (st.hasMoreTokens()) {
                String s = st.nextToken();
                if (s.startsWith("-D")) {
                    String propertyName;
                    String propertyValue;
                    int index = s.indexOf('=');
                    if (index > 0) {
                        propertyName = s.substring(2, index);
                        propertyValue = s.substring(index+1);
                    } else {
                        propertyName = s.substring(2);
                        propertyValue = "";
                    }
                    properties.put(propertyName, propertyValue);
                } else if (s.equals("-cp") || s.equals("-classpath")) {
                    // todo.
                } else if (s.startsWith("-")) {
                    System.err.println("Unsupported option: "+s);
                } else {
                    Class mainClass = Class.forName(s, true, myClassLoader);
                    mainMethod = mainClass.getDeclaredMethod("main", new Class[] { String[].class });
                    List a = new LinkedList();
                    while (st.hasMoreTokens()) {
                        a.add(st.nextToken());
                    }
                    args = (String[]) a.toArray(new String[a.size()]);
                }
            }
        }
        
        public void go() throws IllegalArgumentException, IllegalAccessException, InvocationTargetException {
            Properties old = System.getProperties();
            System.setProperties(properties);
            try {
                mainMethod.invoke(null, new Object[] { args });
            } finally {
                System.setProperties(old);
            }
        }

        
    }
}
