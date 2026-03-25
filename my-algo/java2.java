import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.stream.*;
import java.lang.reflect.*;
import java.io.*;

public class ComplexAnalyzerTest {
    
    // 1. Complex generics with nested wildcards and bounds
    private static class ComplexGeneric<T extends Comparable<? super T> & Serializable> 
        implements Comparable<ComplexGeneric<T>> {
        private T value;
        private List<? extends Map<? super String, ? extends Number>> weirdList;
        
        public ComplexGeneric(T value) {
            this.value = value;
            this.weirdList = new ArrayList<>();
        }
        
        @Override
        public int compareTo(ComplexGeneric<T> other) {
            return value.compareTo(other.value);
        }
        
        public <U extends T> U confusingMethod(List<? super U> list) {
            return (U) value;
        }
    }
    
    // 2. Recursive generic type with multiple interfaces
    private interface Node<T extends Node<T>> extends Cloneable, Serializable {
        T getParent();
        List<T> getChildren();
        T addChild(T child);
    }
    
    private static class TreeNode implements Node<TreeNode> {
        private TreeNode parent;
        private List<TreeNode> children = new ArrayList<>();
        
        @Override
        public TreeNode getParent() { return parent; }
        
        @Override
        public List<TreeNode> getChildren() { return children; }
        
        @Override
        public TreeNode addChild(TreeNode child) {
            child.parent = this;
            children.add(child);
            return this;
        }
    }
    
    // 3. Complex inheritance chain with covariant returns
    static class A {
        Number method() throws IOException { return 1; }
        <T> T genericMethod(T t) { return t; }
    }
    
    static class B extends A {
        @Override
        Integer method() throws FileNotFoundException { return 2; } // Covariant return
        
        @Override
        <T extends Number> T genericMethod(T t) { return t; } // More restrictive bound
    }
    
    static class C extends B {
        @Override
        Integer method() { return 3; } // Throws clause removed
    }
    
    // 4. Multiple interface implementation with same method signature
    interface I1 { void doSomething(); }
    interface I2 { int doSomething(); } // Different return type - potential conflict
    
    static class ConflictClass implements I1, I2 {
        public void doSomething() { } // Implements I1
        
        public int doSomething() { return 42; } // Implements I2 - COMPILE ERROR in real Java
        // This is invalid Java but tests analyzer's type checking
    }
    
    // 5. Complex lambda and method reference expressions
    private static void testLambdas() {
        // Nested lambdas with wildcards
        Function<Function<IntUnaryOperator, String>, Consumer<String>> crazyLambda = 
            (Function<IntUnaryOperator, String> f) -> 
                (String s) -> System.out.println(f.apply(x -> x * 2));
        
        // Method references to overloaded methods
        Consumer<String> c1 = System.out::println;
        Consumer<Integer> c2 = System.out::println;
        
        // Constructor reference with diamond operator
        Supplier<ArrayList<String>> listSupplier = ArrayList::new;
        
        // Lambda capturing effectively final but modified variable
        int[] counter = {0};
        Runnable r = () -> {
            counter[0]++; // Modifying array element, not variable
            System.out.println(counter[0]);
        };
    }
    
    // 6. Complex stream operations with side effects
    private static void testStreams() {
        List<List<String>> nested = Arrays.asList(
            Arrays.asList("a", "b"),
            Arrays.asList("c", "d", "e"),
            Arrays.asList()
        );
        
        // Complex stream pipeline
        Map<Integer, Long> result = nested.stream()
            .filter(list -> !list.isEmpty())
            .flatMap(List::stream)
            .collect(Collectors.groupingBy(
                String::length,
                Collectors.mapping(
                    s -> s.toUpperCase(),
                    Collectors.filtering(
                        s -> s.contains("A"),
                        Collectors.counting()
                    )
                )
            ));
    }
    
    // 7. Reflection with generics and type erasure challenges
    private static void testReflection() throws Exception {
        List<String> stringList = new ArrayList<>();
        stringList.add("test");
        
        // Type erasure makes this dangerous
        Method addMethod = List.class.getMethod("add", Object.class);
        addMethod.invoke(stringList, 42); // Adding Integer to List<String> at runtime!
        
        // Wildcard capture reflection
        List<?> wildcardList = new ArrayList<String>();
        Method clearMethod = List.class.getMethod("clear");
        clearMethod.invoke(wildcardList); // Should work
        
        // Getting generic type parameters
        ParameterizedType type = (ParameterizedType) 
            ComplexGeneric.class.getGenericSuperclass();
        Type[] typeArgs = type.getActualTypeArguments();
    }
    
    // 8. Concurrency issues and race conditions
    private static class RaceCondition {
        private int count = 0;
        private volatile boolean flag = false;
        
        public void increment() {
            // Non-atomic operation - race condition
            count++;
        }
        
        public void unsafeCheckThenAct() {
            if (!flag) {
                // Race condition window here
                try { Thread.sleep(1); } catch (InterruptedException e) {}
                flag = true;
            }
        }
    }
    
    // 9. Complex switch expressions and pattern matching (Java 14+)
    private static String testSwitch(Object obj) {
        return switch (obj) {
            case Integer i when i > 0 -> "Positive integer: " + i;
            case Integer i -> "Non-positive integer: " + i;
            case String s when !s.isEmpty() -> "Non-empty string: " + s;
            case String s -> "Empty string";
            case null -> "Null object";
            case int[] arr when arr.length > 0 -> "Non-empty int array";
            case int[] arr -> "Empty int array";
            default -> "Unknown object: " + obj.getClass();
        };
    }
    
    // 10. Try-with-resources with multiple resources and suppressed exceptions
    private static void testTryWithResources() throws Exception {
        try (
            ByteArrayInputStream in1 = new ByteArrayInputStream(new byte[10]);
            ByteArrayOutputStream out1 = new ByteArrayOutputStream();
            // AutoCloseable that throws exception
            AutoCloseable problematic = () -> { throw new IOException("Close failed"); };
        ) {
            // Normal execution
            in1.read();
            out1.write(1);
            throw new RuntimeException("Main exception");
        } catch (Exception e) {
            // Should have suppressed exceptions from close()
            Throwable[] suppressed = e.getSuppressed();
        }
    }
    
    // 11. Complex enum with abstract methods and constant-specific behavior
    private enum Operation {
        PLUS("+") { 
            double apply(double x, double y) { return x + y; } 
        },
        MINUS("-") { 
            double apply(double x, double y) { return x - y; } 
        },
        TIMES("*") { 
            double apply(double x, double y) { return x * y; } 
        },
        DIVIDE("/") { 
            double apply(double x, double y) { 
                if (y == 0) throw new ArithmeticException("Divide by zero");
                return x / y; 
            } 
        };
        
        private final String symbol;
        
        Operation(String symbol) {
            this.symbol = symbol;
        }
        
        @Override public String toString() { return symbol; }
        
        abstract double apply(double x, double y);
    }
    
    // 12. Varargs with generics and heap pollution potential
    @SafeVarargs
    private static <T> void unsafeVarargs(List<T>... lists) {
        Object[] array = lists; // Valid - array of raw types
        array[0] = Arrays.asList(42); // Heap pollution!
    }
    
    // 13. Complex annotation with retention and target
    @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.TYPE, ElementType.METHOD})
    @interface ComplexAnnotation {
        String value() default "";
        Class<?>[] classes() default {};
        int priority() default 0;
    }
    
    @ComplexAnnotation(
        value = "Test",
        classes = {String.class, Integer.class},
        priority = 1
    )
    private static class AnnotatedClass {
        @Deprecated
        @SuppressWarnings({"unchecked", "rawtypes"})
        public void deprecatedMethod() {
            // Raw type usage
            List list = new ArrayList();
            list.add("test");
        }
    }
    
    // 14. Complex static initialization and circular dependencies
    static class CircularDependency {
        static final int A = B + 1; // Forward reference to B
        static final int B = 10;
        static final int C = A + B; // Depends on both
        
        static {
            // Complex static initializer
            System.out.println("A=" + A + ", B=" + B + ", C=" + C);
        }
    }
    
    // 15. Bridge methods and synthetic methods
    interface GenericInterface<T> {
        T process(T input);
    }
    
    static class StringProcessor implements GenericInterface<String> {
        @Override
        public String process(String input) {
            return input.toUpperCase();
        }
        // Compiler generates bridge method: Object process(Object)
    }
    
    // 16. Complex nested classes with shadowing
    static class Outer {
        private int x = 10;
        
        class Inner {
            private int x = 20; // Shadows Outer.x
            
            void test() {
                int x = 30; // Shadows Inner.x
                System.out.println(x); // 30
                System.out.println(this.x); // 20
                System.out.println(Outer.this.x); // 10
            }
            
            class DeepInner {
                void test() {
                    // Accessing through multiple levels
                    System.out.println(Inner.this.x);
                    System.out.println(Outer.this.x);
                }
            }
        }
    }
    
    // 17. Exception handling with multi-catch and final rethrow
    private static void testExceptions() throws IOException, SQLException {
        try {
            if (Math.random() > 0.5) {
                throw new IOException("IO error");
            } else {
                throw new SQLException("DB error");
            }
        } catch (IOException | SQLException e) { // Multi-catch
            // e is effectively final
            throw e; // Rethrow - precise type is maintained
        }
    }
    
    // 18. Complex ternary operator nesting
    private static int complexTernary(int a, int b, int c) {
        return a > b ? 
               (b > c ? a : c > a ? b : c) : 
               (a > c ? (b > a ? c : b) : (c > b ? a : b));
    }
    
    // 19. Method overloading with varargs and autoboxing
    static class OverloadTest {
        void process(int i) { System.out.println("int"); }
        void process(Integer i) { System.out.println("Integer"); }
        void process(int... i) { System.out.println("varargs int"); }
        void process(Integer... i) { System.out.println("varargs Integer"); }
        void process(Object o) { System.out.println("Object"); }
        void process(Number n) { System.out.println("Number"); }
        
        void test() {
            process(1); // Calls process(int)
            process(Integer.valueOf(1)); // Calls process(Integer)
            process(1, 2); // Calls process(int...)
            process(new Integer[]{1, 2}); // Calls process(Integer...)
            process(null); // AMBIGUOUS - compiler error in real Java
        }
    }
    
    // 20. Resource leak in finally block
    private static void potentialResourceLeak() {
        InputStream in = null;
        try {
            in = new FileInputStream("test.txt");
            // Do something
        } catch (IOException e) {
            // Handle
        } finally {
            // What if close() throws exception?
            if (in != null) {
                try {
                    in.close(); // Exception here masks original exception
                } catch (IOException e) {
                    // Swallowed exception
                }
            }
        }
    }
    
    // 21. Complex array declarations
    private static void arrayDeclarations() {
        int[] a1, a2[]; // a1 is int[], a2 is int[][]
        int[][] b1, b2; // Both are int[][]
        int c1[], c2; // c1 is int[], c2 is int
        
        // Complex array initialization
        int[][][] threeD = new int[3][][];
        for (int i = 0; i < threeD.length; i++) {
            threeD[i] = new int[i + 1][];
            for (int j = 0; j < threeD[i].length; j++) {
                threeD[i][j] = new int[(i + 1) * (j + 1)];
            }
        }
    }
    
    // MAIN METHOD with all test cases
    public static void main(String[] args) throws Exception {
        System.out.println("Testing complex Java patterns...");
        
        // Test various patterns
        testLambdas();
        testStreams();
        testReflection();
        testSwitch(42);
        testTryWithResources();
        testExceptions();
        arrayDeclarations();
        
        // Create instances to test
        ComplexGeneric<String> cg = new ComplexGeneric<>("test");
        TreeNode root = new TreeNode();
        root.addChild(new TreeNode());
        
        // Test circular dependency
        System.out.println("Circular: " + CircularDependency.C);
        
        // Test ternary complexity
        System.out.println("Ternary: " + complexTernary(5, 3, 7));
        
        // Test race condition
        RaceCondition rc = new RaceCondition();
        ExecutorService es = Executors.newFixedThreadPool(10);
        for (int i = 0; i < 1000; i++) {
            es.submit(rc::increment);
        }
        es.shutdown();
        es.awaitTermination(1, TimeUnit.SECONDS);
        
        System.out.println("Test completed (with intentional issues for analyzer detection)");
    }
}