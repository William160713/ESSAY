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
    <title>報酬率篩選</title>
</head>
<body>
    <?php echo "<h1 >報酬率篩選 " . $_SESSION['username'] . "</h1>"; ?>
    <!--
    <div style="text-align:right;"><a href="logout.php">登出</a>
    -->
    <div style="text-align:right;"><input type="button" value="登出"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='logout.php'"></div>
    <div style="text-align:center;">


    <?php echo "<h1 >你選擇的股票是: " . $_POST['stock'] . "</h1>"; ?>
    
    <div style="text-align:center;"><input type="button" value="下一步"  style="width:120px;height:40px;font-size:20px;" onclick="location.href='Strategy.php'"></div>
                         

</body>
</html>