<?php 
header("Content-Type:text/html; charset=big5");
session_start();

if (!isset($_SESSION['username'])) {
    header("Location: index.php");
}


?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>挑選</title>
</head>
<body>
    <?php echo "<h1 >挑選 " . $_SESSION['username'] . "</h1>"; ?>
    <!--
    <div style="text-align:right;"><a href="logout.php">登出</a>
    -->
    
    <div style="text-align:right;"><input type="button" value="登出"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='logout.php'"></div>
    <div style="text-align:center;">
    <div style="text-align:center;">
        



        <br>
        <br>
        <br>
        <h1>選擇你適合的投資組合:</h1>
        

      <form action="Rate_of_return.php" method="post">
        <table border="1" width="800"  align="center">
        <tr>
            <td>保本型投資</td>
            <td>平衡型投資</td>
            <td>成長型投資</td>
        </tr>
        <tr>
            <td><input type="radio" name="stock" value="0050">0050</td>
            <td><input type="radio" name="stock" value="2330">2330</td>
            <td><input type="radio" name="stock" value="2615">2615</td>
        </tr>
        

          </table>    

    </div>

    

    <br>
    <br>
    <br>
    
　
　  
    



    <div style="text-align:center;"><input type="submit" value="下一步"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='Rate_of_return.php'"></div>
                         
    </form>
</body>
</html>